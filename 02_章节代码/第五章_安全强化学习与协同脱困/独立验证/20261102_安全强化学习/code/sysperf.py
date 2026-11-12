"""Process settings for long CPU runs on this Windows 11 machine (README §6, resources).

Windows 11 applies EcoQoS execution-speed throttling to background console processes; on 2026-11-02 one control
sample of filter L (N = 8, B = 2) took about 355 ms throttled and 42 ms unthrottled. Each process therefore opts out
of execution-speed throttling, and lowers its own priority so that interactive programs keep precedence.
Neither setting changes any computed number."""
import ctypes
import sys
from ctypes import wintypes


class _PPTS(ctypes.Structure):
    _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]


def configure():
    if sys.platform != 'win32':
        return dict(throttling_disabled=None, below_normal=None)
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    h = k32.GetCurrentProcess()
    st = _PPTS(1, 0x1, 0x0)            # PROCESS_POWER_THROTTLING_EXECUTION_SPEED: controlled, not throttled
    ok1 = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))   # 4 = ProcessPowerThrottling
    ok2 = bool(k32.SetPriorityClass(h, 0x00004000))                                    # BELOW_NORMAL_PRIORITY_CLASS
    return dict(throttling_disabled=ok1, below_normal=ok2)
