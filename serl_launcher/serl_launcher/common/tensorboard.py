import numpy as np
import concurrent.futures
import ml_collections
import datetime
import tensorflow as tf
 
def Set_GPU_Memory_Growth():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# 设置 GPU 显存占用为按需分配
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# 异常处理
			print(e)
	else :
		print('No GPU')
 
# 放在建立模型实例之前
Set_GPU_Memory_Growth()


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values

class AsyncOutput:

  def __init__(self, callback, parallel=True):
    self._callback = callback
    self._parallel = parallel
    if parallel:
      self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._future = None

  def __call__(self, summaries):
    if self._parallel:
      self._future and self._future.result()
      self._future = self._executor.submit(self._callback, summaries)
    else:
      self._callback(summaries)


class TensorBoardLogger(AsyncOutput):
    
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(self, config, logdir, fps=20, maxsize=1e9, parallel=True):
        super().__init__(self.log, parallel)
        self.config = config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)
        self._logdir = str(f"{logdir}/{self.config.exp_descriptor}_{self.config.unique_identifier}")
        self._fps = fps
        self._writer = None
        self._maxsize = maxsize
        if self._maxsize:
            self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=3)
            self._promise = None

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        reset = False
        if self._maxsize:
            result = self._promise and self._promise.result()
            # print('Current TensorBoard event file size:', result)
            reset = (self._promise and result >= self._maxsize)
            self._promise = self._checker.submit(self._check)
        if not self._writer or reset:
            print('Creating new TensorBoard event file writer.')
            self._writer = tf.summary.create_file_writer(
                self._logdir, flush_millis=1000, max_queue=10000)
            self._writer.set_as_default()
        for name, value in data.items():
            try:
                if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float): 
                    if isinstance(value, bool):
                        value = np.array(int(value))
                    else:
                        value = np.array(value)
                if len(value.shape) == 0:
                    tf.summary.scalar(name, value, step)
                elif len(value.shape) == 1:
                    if len(value) > 1024:
                        value = value.copy()
                        np.random.shuffle(value)
                        value = value[:1024]
                    tf.summary.histogram(name, value, step)
                elif len(value.shape) == 2:
                    tf.summary.image(name, value, step)
                elif len(value.shape) == 3:
                    tf.summary.image(name, value, step)
                elif len(value.shape) == 4:
                    self._video_summary(name, value, step)
            except Exception:
                print('Error writing summary:', name)
                raise
        self._writer.flush()
    def _check(self):
        events = tf.io.gfile.glob(self._logdir.rstrip('/') + '/events.out.*')
        return tf.io.gfile.stat(sorted(events)[-1]).length if events else 0

    def _video_summary(self, name, video, step):
        import tensorflow.compat.v1 as tf1
        name = name if isinstance(name, str) else name.decode('utf-8')
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = _encode_gif(video, self._fps)
            summary.value.add(tag=name, image=image)
            tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
        except (IOError, OSError) as e:
            print('GIF summaries require ffmpeg in $PATH.', e)
            tf.summary.image(name, video, step)
            
            
def _encode_gif(frames, fps):
  # Don't need to use ffmpeg, just use imageio
  import imageio
  import io
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  images = [np.array(frame) for frame in frames]
  with io.BytesIO() as buffer:
      imageio.mimwrite(
          buffer,
          frames,
          duration=(1000 * 1/fps),
          palettesize=256,
          format='GIF',
          loop=0
      )
      binary_gif = buffer.getvalue()

  return binary_gif