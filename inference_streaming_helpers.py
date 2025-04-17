"""
install some of my personal favorites first:
pip install typer methodtools loguru httpx
"""

import os, sys, typer, videoseal, torch, ffmpeg, subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd
from methodtools import lru_cache
from loguru import logger
from inference_streaming import detect_video_clip, embed_video_clip

app = typer.Typer(pretty_exceptions_show_locals=False)
logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)


def is_valid_url(url_string):
    import httpx

    try:
        url = httpx.URL(url_string)
        return url.scheme and url.host
    except Exception:
        return False


@lru_cache()
def load_model():
    # Device selection priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    video_model = videoseal.load("videoseal")
    video_model.eval()
    video_model.to(device)
    video_model.compile()
    logger.info(f"Using device: {device}")
    return video_model


#########################
# binary-string helpers
#########################


def string_to_bin8(in_str: str):
    """returns a list of 8-bit binary strings (one for each character)"""
    return [format(ord(char), "08b") for char in in_str]


def bin8_to_string(bin8_str: str, bits_per_char: int = 8):
    binary_values = [
        bin8_str[i : i + bits_per_char] for i in range(0, len(bin8_str), bits_per_char)
    ]  # Split into 8-bit chunks
    original_string = "".join(
        [chr(int(b, base=2)) for b in binary_values]
    )  # Convert each chunk to a character
    return original_string


@app.command()
def string_to_tensor(
    input_string: str, tensor_len: int = 256, debug: bool = False, fill_char: str = " "
):
    # compute bytes available from tensor_len
    bytes_avail = tensor_len / 8
    if len(input_string) > bytes_avail:
        logger.warning(
            f"input_string requires {len(input_string)} bytes but only {bytes_avail} bytes are available, your string will be truncated!"
        )

    # Pad or truncate to available bytes
    input_string = input_string.ljust(int(bytes_avail), fill_char)

    # Encode the string to bytes
    byte_data = string_to_bin8(input_string)

    # Convert bytes to a list of integers
    int_data = [int(i) for i in "".join(byte_data)]

    # Create a tensor from the integer list
    tensor = torch.tensor(int_data, dtype=torch.uint8)

    if debug:
        logger.debug(f"{input_string} encoded to tensor:\n{tensor}")
    return tensor


def tensor_to_string(tensor):
    binary_string = "".join([str(i) for i in tensor.numpy().tolist()])
    return bin8_to_string(binary_string).rstrip()


@app.command()
def load_tensor_msg(txt_path: str, expected_len: int = 96, to_string: bool = False):
    # the message should ideally be exported using torch.save()
    # and imported again with torch.load()
    with open(txt_path, "r") as f:
        content = f.read()

    # Convert the string back to a list using eval
    loaded_list = [int(i) for i in content]
    if len(loaded_list) != expected_len:
        logger.warning(
            f"tensor in {txt_path} is of length {len(loaded_list)}, expected {expected_len}"
        )

    # Convert the list to a PyTorch tensor
    loaded_tensor = torch.tensor(loaded_list)
    msg = tensor_to_string(loaded_tensor) if to_string else loaded_tensor

    logger.debug(f"loaded tensor message: \n{msg}")
    return msg


@app.command()
def detect_video(
    input_path: str,
    chunk_size: int,
    model=None,
    max_frames: int = None,
    tqdm_func=tqdm,
) -> None:
    model = model if model else load_model()
    # Read video dimensions
    probe = ffmpeg.probe(input_path)
    video_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])
    codec = video_info["codec_name"]
    num_frames = int(probe["streams"][0]["nb_frames"])

    # Open the input video
    process1 = (
        ffmpeg.input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frames_in_chunk = 0
    frames_count = 0
    max_frames = max_frames if max_frames else num_frames
    soft_msgs = []
    for in_bytes in tqdm_func(
        iter(lambda: process1.stdout.read(frame_size), b""),
        total=num_frames,
        desc="Watermark extraction",
        position=1,
    ):
        # Convert bytes to frame and add to chunk
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frames_in_chunk] = frame
        frames_in_chunk += 1

        # Process chunk when full
        if frames_in_chunk == chunk_size:
            soft_msgs.append(detect_video_clip(model, chunk))
            frames_in_chunk = 0

        # Count number of processed frames
        frames_count += 1
        if frames_count > max_frames:
            logger.debug(f"extraction early stop at frame {frames_count}")
            break

    # Process final partial chunk if any
    if frames_in_chunk > 0:
        soft_msgs.append(detect_video_clip(model, chunk))

    process1.stdout.close()
    process1.wait()

    soft_msgs = torch.cat(soft_msgs, dim=0)
    soft_msgs = soft_msgs.mean(dim=0)  # Average the predictions across all frames
    return soft_msgs


@app.command()
def get_signature(
    vid_path: str,
    chunk_size: int = 16,
    signature_txt_path: str = None,
    max_frames: int = 0,
    tqdm_override=None,
):
    tqdm_func = tqdm_override if tqdm_override else tqdm
    m = load_model()
    soft_msgs = detect_video(
        model=m,
        input_path=vid_path,
        chunk_size=chunk_size,
        max_frames=max_frames,
        tqdm_func=tqdm_func,
    )

    if signature_txt_path:
        from videoseal.evals.metrics import bit_accuracy

        target_msg = load_tensor_msg(signature_txt_path)
        bit_acc = bit_accuracy(soft_msgs, target_msg).item() * 100
        logger.success(f"Binary message extracted with {bit_acc:.1f}% bit accuracy")

    # Binarize Float Tensor
    soft_msgs = (soft_msgs > 0).int()
    logger.info(f"soft message:\n{soft_msgs}")

    sig = tensor_to_string(soft_msgs)
    logger.info(f"decoded signiture:\n{sig}")
    return {"soft_message": soft_msgs, "signature": sig}


@app.command()
def embed_video(
    input_path: str,
    output_path: str,
    msg: str,
    chunk_size: int = 16,
    keep_audio: bool = True,
    save_msg: bool = True,
    # model: Videoseal
) -> None:

    # input validation
    assert output_path.endswith(".mp4"), f"output_path needs end with .mp4"

    # load Video Seal model
    model = load_model()

    # Read video dimensions
    probe = ffmpeg.probe(input_path)
    video_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width = int(video_info["width"])
    height = int(video_info["height"])
    fps = float(video_info["r_frame_rate"].split("/")[0]) / float(
        video_info["r_frame_rate"].split("/")[1]
    )
    codec = video_info["codec_name"]
    num_frames = int(probe["streams"][0]["nb_frames"])

    # Open the input video
    process1 = (
        ffmpeg.input(input_path)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=fps,
        )
        .run_async(pipe_stdout=True, pipe_stderr=subprocess.PIPE)
    )
    # Open the output video with optimal thread usage.
    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=fps,
        )
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=subprocess.PIPE)
    )

    # Convert message to tensor
    soft_msg = string_to_tensor(msg)
    # Add a new dimension at axis 0 to make the shape [1, 256]
    soft_msg = torch.unsqueeze(soft_msg, dim=0)
    if save_msg:
        with open(output_path.replace(".mp4", ".txt"), "w") as f:
            f.write("".join([str(i.item()) for i in soft_msg[0]]))
        logger.info(
            f'signature tensor created from "{msg}": {output_path.replace(".mp4", ".txt")}'
        )

    # Process the video
    frame_size = width * height * 3
    chunk = np.zeros((chunk_size, height, width, 3), dtype=np.uint8)
    frames_in_chunk = 0

    for in_bytes in tqdm(
        iter(lambda: process1.stdout.read(frame_size), b""),
        total=num_frames,
        desc="Watermark embedding",
        position=1,
    ):
        # Convert bytes to frame and add to chunk
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        chunk[frames_in_chunk] = frame
        frames_in_chunk += 1

        # Process chunk when full
        if frames_in_chunk == chunk_size:
            # print(f"embedding at frame: {frame_idx}")
            processed_frames = embed_video_clip(model, chunk, soft_msg)
            process2.stdin.write(processed_frames.tobytes())
            frames_in_chunk = 0

    # Process final partial chunk if any
    if frames_in_chunk > 0:
        processed_frames = embed_video_clip(model, chunk[:frames_in_chunk], soft_msg)
        process2.stdin.write(processed_frames.tobytes())

    process1.stdout.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()

    # Preserve Audio
    if keep_audio:
        # Copy just the audio from the original video
        temp_output = output_path + ".tmp"
        os.rename(output_path, temp_output)

        audiostream = ffmpeg.input(input_path)
        videostream = ffmpeg.input(temp_output)
        process3 = (
            ffmpeg.output(
                videostream.video,
                audiostream.audio,
                output_path,
                vcodec="copy",
                acodec="copy",
            )
            .overwrite_output()
            .run_async(pipe_stderr=subprocess.PIPE)
        )
        process3.wait()
        os.remove(temp_output)
        logger.debug("Copied audio from the original video")

    logger.info(f"signed video created: {output_path}")
    return soft_msg


@app.command()
def embed_videos_df(
    dataframe,
    output_dir: str,
    signature_col: str = "media_id",
    video_path_col: str = "hq_download_link",
    signature_prefix: str = "gf:",
    overwrite: bool = False,
    save_soft_msg: bool = False,
):
    df = dataframe if isinstance(dataframe, pd.DataFrame) else pd.read_csv(dataframe)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="signing videos", position=0):
        # if not is_valid_url(row[video_path_col]) or not os.path.isfile(row[video_path_col]):
        if not isinstance(row[video_path_col], str):
            logger.debug(f"missing {video_path_col} at row {i}")
            continue

        vname, vext = os.path.splitext(os.path.basename(row[video_path_col]))
        vext = vext if vext else ".mp4"
        out_fp = os.path.join(output_dir, vname + vext)
        if overwrite or not os.path.isfile(out_fp):
            embed_video(
                row[video_path_col],
                output_path=out_fp,
                msg=f"{signature_prefix}{row[signature_col]}",
                keep_audio=True,
                save_msg=save_soft_msg,
            )
        else:
            logger.debug(f"{out_fp} already exist")


if __name__ == "__main__":
    app()
