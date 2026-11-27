"""Minimal headless libretro frontend in pure ctypes.

No X11, no SDL, no RetroArch: the core is a shared object, we hand it
callbacks, and we drive it one frame at a time with retro_run().

One core instance per PROCESS. Libretro cores keep global state, so two
LibretroCore objects in one process will corrupt each other. Parallelism
comes from processes (see vec.py).
"""

import ctypes
import os
from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, byref, c_bool,
                    c_char_p, c_double, c_float, c_int, c_int16, c_size_t,
                    c_uint, c_uint8, c_uint32, c_void_p)

import numpy as np

# ---------------------------------------------------------------- constants

EXPERIMENTAL = 0x10000

ENV_SET_ROTATION = 1
ENV_GET_OVERSCAN = 2
ENV_GET_CAN_DUPE = 3
ENV_SET_MESSAGE = 6
ENV_SHUTDOWN = 7
ENV_SET_PERFORMANCE_LEVEL = 8
ENV_GET_SYSTEM_DIRECTORY = 9
ENV_SET_PIXEL_FORMAT = 10
ENV_SET_INPUT_DESCRIPTORS = 11
ENV_SET_KEYBOARD_CALLBACK = 12
ENV_SET_DISK_CONTROL_INTERFACE = 13
ENV_SET_HW_RENDER = 14
ENV_GET_VARIABLE = 15
ENV_SET_VARIABLES = 16
ENV_GET_VARIABLE_UPDATE = 17
ENV_SET_SUPPORT_NO_GAME = 18
ENV_GET_LIBRETRO_PATH = 19
ENV_SET_FRAME_TIME_CALLBACK = 21
ENV_SET_AUDIO_CALLBACK = 22
ENV_GET_RUMBLE_INTERFACE = 23
ENV_GET_INPUT_DEVICE_CAPABILITIES = 24
ENV_GET_SENSOR_INTERFACE = 25 | EXPERIMENTAL
ENV_GET_CAMERA_INTERFACE = 26 | EXPERIMENTAL
ENV_GET_LOG_INTERFACE = 27
ENV_GET_PERF_INTERFACE = 28
ENV_GET_LOCATION_INTERFACE = 29
ENV_GET_CORE_ASSETS_DIRECTORY = 30
ENV_GET_SAVE_DIRECTORY = 31
ENV_SET_SYSTEM_AV_INFO = 32
ENV_SET_PROC_ADDRESS_CALLBACK = 33
ENV_SET_SUBSYSTEM_INFO = 34
ENV_SET_CONTROLLER_INFO = 35
ENV_SET_MEMORY_MAPS = 36 | EXPERIMENTAL
ENV_SET_GEOMETRY = 37
ENV_GET_USERNAME = 38
ENV_GET_LANGUAGE = 39
ENV_GET_CURRENT_SOFTWARE_FRAMEBUFFER = 40 | EXPERIMENTAL
ENV_GET_HW_RENDER_INTERFACE = 41 | EXPERIMENTAL
ENV_SET_SUPPORT_ACHIEVEMENTS = 42 | EXPERIMENTAL
ENV_GET_VFS_INTERFACE = 45 | EXPERIMENTAL
ENV_GET_LED_INTERFACE = 46 | EXPERIMENTAL
ENV_GET_AUDIO_VIDEO_ENABLE = 47
ENV_GET_MIDI_INTERFACE = 48 | EXPERIMENTAL
ENV_GET_FASTFORWARDING = 49 | EXPERIMENTAL
ENV_GET_TARGET_REFRESH_RATE = 50 | EXPERIMENTAL
ENV_GET_INPUT_BITMASKS = 51 | EXPERIMENTAL
ENV_GET_CORE_OPTIONS_VERSION = 52
ENV_SET_CORE_OPTIONS = 53
ENV_SET_CORE_OPTIONS_INTL = 54
ENV_SET_CORE_OPTIONS_DISPLAY = 55
ENV_GET_DISK_CONTROL_INTERFACE_VERSION = 57
ENV_SET_DISK_CONTROL_EXT_INTERFACE = 58
ENV_GET_MESSAGE_INTERFACE_VERSION = 59
ENV_GET_INPUT_MAX_USERS = 61
ENV_SET_AUDIO_BUFFER_STATUS_CALLBACK = 62
ENV_SET_MINIMUM_AUDIO_LATENCY = 63
ENV_SET_CORE_OPTIONS_V2 = 67
ENV_SET_CORE_OPTIONS_V2_INTL = 68
ENV_SET_VARIABLE = 70
ENV_GET_THROTTLE_STATE = 71 | EXPERIMENTAL
ENV_GET_SAVESTATE_CONTEXT = 72 | EXPERIMENTAL
ENV_GET_JIT_CAPABLE = 74
ENV_GET_MICROPHONE_INTERFACE = 75 | EXPERIMENTAL
ENV_GET_DEVICE_POWER = 77 | EXPERIMENTAL

PIXEL_0RGB1555 = 0
PIXEL_XRGB8888 = 1
PIXEL_RGB565 = 2

MEMORY_SAVE_RAM = 0
MEMORY_RTC = 1
MEMORY_SYSTEM_RAM = 2
MEMORY_VIDEO_RAM = 3

DEVICE_JOYPAD = 1
DEVICE_ID_JOYPAD_MASK = 256

# RETRO_DEVICE_ID_JOYPAD_* -> bit position
JOYPAD = {
    "B": 0, "Y": 1, "SELECT": 2, "START": 3,
    "UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7,
    "A": 8, "X": 9, "L": 10, "R": 11,
    "L2": 12, "R2": 13, "L3": 14, "R3": 15,
}
# PlayStation face-button names, as a player would say them.
PSX_BUTTON = {
    "cross": "B", "circle": "A", "square": "Y", "triangle": "X",
    "l1": "L", "r1": "R", "l2": "L2", "r2": "R2",
    "start": "START", "select": "SELECT",
    "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
}

# ---------------------------------------------------------------- structs


class RetroSystemInfo(Structure):
    _fields_ = [("library_name", c_char_p), ("library_version", c_char_p),
                ("valid_extensions", c_char_p), ("need_fullpath", c_bool),
                ("block_extract", c_bool)]


class RetroGameGeometry(Structure):
    _fields_ = [("base_width", c_uint), ("base_height", c_uint),
                ("max_width", c_uint), ("max_height", c_uint),
                ("aspect_ratio", c_float)]


class RetroSystemTiming(Structure):
    _fields_ = [("fps", c_double), ("sample_rate", c_double)]


class RetroSystemAvInfo(Structure):
    _fields_ = [("geometry", RetroGameGeometry), ("timing", RetroSystemTiming)]


class RetroGameInfo(Structure):
    _fields_ = [("path", c_char_p), ("data", c_void_p),
                ("size", c_size_t), ("meta", c_char_p)]


class RetroVariable(Structure):
    _fields_ = [("key", c_char_p), ("value", c_char_p)]


class RetroLogCallback(Structure):
    _fields_ = [("log", c_void_p)]


# ---------------------------------------------------------------- callbacks

ENV_CB = CFUNCTYPE(c_bool, c_uint, c_void_p)
VIDEO_CB = CFUNCTYPE(None, c_void_p, c_uint, c_uint, c_size_t)
AUDIO_CB = CFUNCTYPE(None, c_int16, c_int16)
AUDIO_BATCH_CB = CFUNCTYPE(c_size_t, c_void_p, c_size_t)
INPUT_POLL_CB = CFUNCTYPE(None)
INPUT_STATE_CB = CFUNCTYPE(c_int16, c_uint, c_uint, c_uint, c_uint)
LOG_CB = CFUNCTYPE(None, c_int, c_char_p)


class LibretroError(RuntimeError):
    pass


class LibretroCore:
    """Drives a libretro core frame by frame, with no display server."""

    def __init__(self, core_path, system_dir, save_dir=None, options=None,
                 enable_audio=False, verbose=False, max_users=1):
        self.core_path = os.path.abspath(core_path)
        self.system_dir = os.path.abspath(system_dir).encode()
        self.save_dir = os.path.abspath(save_dir or system_dir).encode()
        self.options = {k.encode(): str(v).encode()
                        for k, v in (options or {}).items()}
        self.enable_audio = enable_audio
        self.verbose = verbose
        self.max_users = max_users

        # RTLD_LOCAL (the default) keeps each core's symbols out of the global
        # namespace; harmless here but required if you ever dlopen two cores.
        self.dll = CDLL(self.core_path)
        self._bind()

        api = self.dll.retro_api_version()
        if api != 1:
            raise LibretroError(f"unsupported libretro API version {api}")

        self.frame = None          # last non-duplicate frame, RGB uint8 HxWx3
        self.frame_is_new = False
        self._convert = True       # see run(convert=...)
        self.width = 0
        self.height = 0
        self.pixel_format = PIXEL_RGB565
        self.av_info = RetroSystemAvInfo()
        self._options_dirty = True
        self._input = np.zeros((max_users, 16), dtype=np.int16)
        self._game_loaded = False
        self._keepalive = []       # ctypes callbacks must outlive the core

        self._install_callbacks()
        self.dll.retro_init()

        info = RetroSystemInfo()
        self.dll.retro_get_system_info(byref(info))
        self.system_info = {
            "name": info.library_name.decode(),
            "version": info.library_version.decode(),
            "extensions": info.valid_extensions.decode(),
            "need_fullpath": bool(info.need_fullpath),
        }

    # ------------------------------------------------------------ plumbing

    def _bind(self):
        d = self.dll
        d.retro_api_version.restype = c_uint
        d.retro_init.restype = None
        d.retro_deinit.restype = None
        d.retro_run.restype = None
        d.retro_reset.restype = None
        d.retro_unload_game.restype = None
        d.retro_get_system_info.argtypes = [POINTER(RetroSystemInfo)]
        d.retro_get_system_av_info.argtypes = [POINTER(RetroSystemAvInfo)]
        d.retro_load_game.argtypes = [POINTER(RetroGameInfo)]
        d.retro_load_game.restype = c_bool
        d.retro_serialize_size.restype = c_size_t
        d.retro_serialize.argtypes = [c_void_p, c_size_t]
        d.retro_serialize.restype = c_bool
        d.retro_unserialize.argtypes = [c_void_p, c_size_t]
        d.retro_unserialize.restype = c_bool
        d.retro_get_memory_data.argtypes = [c_uint]
        d.retro_get_memory_data.restype = c_void_p
        d.retro_get_memory_size.argtypes = [c_uint]
        d.retro_get_memory_size.restype = c_size_t
        d.retro_set_controller_port_device.argtypes = [c_uint, c_uint]
        d.retro_set_controller_port_device.restype = None
        for setter in ("environment", "video_refresh", "audio_sample",
                       "audio_sample_batch", "input_poll", "input_state"):
            getattr(d, "retro_set_" + setter).restype = None

    def _install_callbacks(self):
        self._cb_env = ENV_CB(self._on_env)
        self._cb_video = VIDEO_CB(self._on_video)
        self._cb_audio = AUDIO_CB(lambda l, r: None)
        self._cb_audio_batch = AUDIO_BATCH_CB(lambda data, frames: frames)
        self._cb_poll = INPUT_POLL_CB(lambda: None)
        self._cb_input = INPUT_STATE_CB(self._on_input)
        self._cb_log = LOG_CB(self._on_log)
        self._keepalive += [self._cb_env, self._cb_video, self._cb_audio,
                            self._cb_audio_batch, self._cb_poll,
                            self._cb_input, self._cb_log]
        d = self.dll
        d.retro_set_environment(self._cb_env)
        d.retro_set_video_refresh(self._cb_video)
        d.retro_set_audio_sample(self._cb_audio)
        d.retro_set_audio_sample_batch(self._cb_audio_batch)
        d.retro_set_input_poll(self._cb_poll)
        d.retro_set_input_state(self._cb_input)

    def _on_log(self, level, fmt):
        if self.verbose and fmt:
            try:
                print("[core]", fmt.decode(errors="replace").rstrip())
            except Exception:
                pass

    def _on_env(self, cmd, data):
        if cmd == ENV_GET_SYSTEM_DIRECTORY:
            ctypes.cast(data, POINTER(c_char_p))[0] = self.system_dir
            return True
        if cmd in (ENV_GET_SAVE_DIRECTORY, ENV_GET_CORE_ASSETS_DIRECTORY):
            ctypes.cast(data, POINTER(c_char_p))[0] = self.save_dir
            return True
        if cmd == ENV_SET_PIXEL_FORMAT:
            fmt = ctypes.cast(data, POINTER(c_int))[0]
            if fmt not in (PIXEL_0RGB1555, PIXEL_XRGB8888, PIXEL_RGB565):
                return False
            self.pixel_format = fmt
            return True
        if cmd == ENV_GET_CAN_DUPE:
            ctypes.cast(data, POINTER(c_bool))[0] = True
            return True
        if cmd == ENV_GET_VARIABLE:
            var = ctypes.cast(data, POINTER(RetroVariable))
            value = self.options.get(var[0].key)
            if value is None:
                return False       # let the core keep its own default
            var[0].value = value
            return True
        if cmd == ENV_GET_VARIABLE_UPDATE:
            ctypes.cast(data, POINTER(c_bool))[0] = self._options_dirty
            self._options_dirty = False
            return True
        if cmd == ENV_SET_VARIABLES:
            return True
        if cmd == ENV_GET_CORE_OPTIONS_VERSION:
            # Report v0 so the core falls back to plain SET_VARIABLES and we
            # never have to walk the v2 option structs.
            ctypes.cast(data, POINTER(c_uint))[0] = 0
            return True
        if cmd == ENV_GET_LOG_INTERFACE:
            # Declining this leaves the core logging to stderr. A ctypes
            # callback cannot take printf varargs, so our own handler would
            # only ever print the bare format string.
            return False
        if cmd == ENV_GET_AUDIO_VIDEO_ENABLE:
            # bit0 video, bit1 audio, bit2 fast savestates, bit3 hard-disable
            # audio. Killing audio outright is the single biggest headless win.
            value = 1 | (2 if self.enable_audio else 8)
            ctypes.cast(data, POINTER(c_int))[0] = value
            return True
        if cmd == ENV_GET_INPUT_BITMASKS:
            return True
        if cmd == ENV_GET_INPUT_MAX_USERS:
            ctypes.cast(data, POINTER(c_uint))[0] = self.max_users
            return True
        if cmd in (ENV_SET_SYSTEM_AV_INFO, ENV_SET_GEOMETRY):
            # PSX games switch resolution mid-game; just accept it.
            return True
        if cmd == ENV_GET_LANGUAGE:
            ctypes.cast(data, POINTER(c_uint))[0] = 0   # English
            return True
        if cmd == ENV_GET_OVERSCAN:
            ctypes.cast(data, POINTER(c_bool))[0] = False
            return True
        if cmd == ENV_GET_FASTFORWARDING:
            ctypes.cast(data, POINTER(c_bool))[0] = False
            return True
        if cmd == ENV_GET_TARGET_REFRESH_RATE:
            ctypes.cast(data, POINTER(c_float))[0] = 60.0
            return True
        if cmd == ENV_GET_JIT_CAPABLE:
            ctypes.cast(data, POINTER(c_bool))[0] = True
            return True
        if cmd in (ENV_SET_INPUT_DESCRIPTORS, ENV_SET_CONTROLLER_INFO,
                   ENV_SET_PERFORMANCE_LEVEL, ENV_SET_MEMORY_MAPS,
                   ENV_SET_SUPPORT_ACHIEVEMENTS, ENV_SET_SUBSYSTEM_INFO,
                   ENV_SET_DISK_CONTROL_INTERFACE,
                   ENV_SET_MINIMUM_AUDIO_LATENCY, ENV_SET_ROTATION):
            return True
        if cmd == ENV_SHUTDOWN:
            self.shutdown_requested = True
            return True
        return False               # unknown/unsupported: cores cope fine

    def _on_video(self, data, width, height, pitch):
        self.width, self.height = width, height
        if not self._convert:
            # Caller does not want this frame: skip the colour conversion
            # entirely. Matters inside a frameskip window, where only the
            # last frame is ever observed.
            self.frame_is_new = False
            return
        if not data or width == 0 or height == 0:
            self.frame_is_new = False   # duplicate frame: keep the old one
            return
        self.frame = self._to_rgb(data, width, height, pitch)
        self.frame_is_new = True

    def _to_rgb(self, data, width, height, pitch):
        if self.pixel_format == PIXEL_XRGB8888:
            buf = ctypes.string_at(data, pitch * height)
            arr = np.frombuffer(buf, dtype=np.uint32).reshape(
                height, pitch // 4)[:, :width]
            rgba = arr.view(np.uint8).reshape(height, width, 4)
            return np.ascontiguousarray(rgba[:, :, 2::-1])   # BGRX -> RGB
        buf = ctypes.string_at(data, pitch * height)
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(
            height, pitch // 2)[:, :width].astype(np.uint16)
        out = np.empty((height, width, 3), dtype=np.uint8)
        if self.pixel_format == PIXEL_RGB565:
            out[:, :, 0] = ((arr >> 11) & 0x1F) << 3
            out[:, :, 1] = ((arr >> 5) & 0x3F) << 2
            out[:, :, 2] = (arr & 0x1F) << 3
        else:                                                # 0RGB1555
            out[:, :, 0] = ((arr >> 10) & 0x1F) << 3
            out[:, :, 1] = ((arr >> 5) & 0x1F) << 3
            out[:, :, 2] = (arr & 0x1F) << 3
        return out

    def _on_input(self, port, device, index, id_):
        if port >= self.max_users or device != DEVICE_JOYPAD:
            return 0
        if id_ == DEVICE_ID_JOYPAD_MASK:
            bits = 0
            row = self._input[port]
            for i in range(16):
                if row[i]:
                    bits |= 1 << i
            return bits
        if id_ < 16:
            return int(self._input[port][id_])
        return 0

    # ------------------------------------------------------------- lifecycle

    def load_game(self, path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        info = RetroGameInfo()
        info.path = path.encode()
        info.meta = None
        if self.system_info["need_fullpath"]:
            info.data, info.size = None, 0
        else:
            with open(path, "rb") as fh:
                blob = fh.read()
            self._blob = ctypes.create_string_buffer(blob, len(blob))
            info.data = ctypes.cast(self._blob, c_void_p)
            info.size = len(blob)
        if not self.dll.retro_load_game(byref(info)):
            raise LibretroError(f"core refused to load {path}")
        self._game_loaded = True
        self.dll.retro_get_system_av_info(byref(self.av_info))
        self.dll.retro_set_controller_port_device(0, DEVICE_JOYPAD)
        return {
            "base_width": self.av_info.geometry.base_width,
            "base_height": self.av_info.geometry.base_height,
            "max_width": self.av_info.geometry.max_width,
            "max_height": self.av_info.geometry.max_height,
            "fps": self.av_info.timing.fps,
            "sample_rate": self.av_info.timing.sample_rate,
        }

    def set_buttons(self, buttons, port=0):
        """buttons: iterable of libretro joypad names, or a 16-bit int mask."""
        row = self._input[port]
        row[:] = 0
        if isinstance(buttons, int):
            for i in range(16):
                row[i] = (buttons >> i) & 1
            return
        for name in buttons:
            key = name.upper()
            if key not in JOYPAD:
                key = PSX_BUTTON.get(name.lower())
                if key is None:
                    raise KeyError(f"unknown button {name!r}")
            row[JOYPAD[key]] = 1

    def run(self, convert=True):
        """Advance exactly one emulated frame.

        convert=False skips turning the framebuffer into an RGB array, for
        frames the caller will never look at. Returns the current frame,
        which is stale when convert=False.
        """
        self._convert = convert
        self.dll.retro_run()
        return self.frame

    def reset(self):
        self.dll.retro_reset()

    # ---------------------------------------------------------------- memory

    def memory(self, kind=MEMORY_SYSTEM_RAM):
        """Zero-copy uint8 view of core memory (PS1 main RAM is 2 MiB)."""
        size = self.dll.retro_get_memory_size(kind)
        ptr = self.dll.retro_get_memory_data(kind)
        if not ptr or size == 0:
            return None
        buf = (c_uint8 * size).from_address(ptr)
        return np.ctypeslib.as_array(buf)

    # ------------------------------------------------------------ savestates

    def serialize(self):
        size = self.dll.retro_serialize_size()
        if size == 0:
            raise LibretroError("core reports savestate size 0")
        buf = ctypes.create_string_buffer(size)
        if not self.dll.retro_serialize(ctypes.cast(buf, c_void_p), size):
            raise LibretroError("retro_serialize failed")
        return buf.raw

    def unserialize(self, blob):
        buf = ctypes.create_string_buffer(blob, len(blob))
        if not self.dll.retro_unserialize(ctypes.cast(buf, c_void_p), len(blob)):
            raise LibretroError("retro_unserialize failed")

    def close(self):
        if getattr(self, "_game_loaded", False):
            self.dll.retro_unload_game()
            self._game_loaded = False
        if getattr(self, "dll", None) is not None:
            self.dll.retro_deinit()
            self.dll = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
