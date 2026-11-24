"""子进程控制台隔离：harness 派生的任何子进程都不得与 TUI 共用控制台。

Windows 上控制台是「一个宿主进程 + N 个附着的客户端」，输入缓冲、输入/输出模式
都是**整个控制台的全局状态**。子进程默认继承父进程的控制台——即使它的 stdout/stderr
已经重定向到管道，它（和它的全部后代）仍然附着在 TUI 的控制台上，可以：

- 改控制台模式：MSYS2/cygwin 运行时在跑原生子进程期间会改写模式，git 启动编辑器前后
  也会 save/restore。TUI 的渲染线程在同一时刻往同一个控制台写，承受全部后果。
- 读走键盘输入：cygwin 的 cons_master_thread 会把输入记录读出再写回；vim 这类
  交互程序直接抢键盘，TUI 的 Esc/插话收不到。

2026-11-13 真机事故（会话 b379c5b3aaa0）：一条 `git rebase -i`（bash → git.exe → sh
编辑器脚本）期间控制台输出模式被改，Rich Live 恰好写入双宽中文字符，命中 Windows
Terminal 1.24 的宿主死循环（microsoft/terminal#20693）——OpenConsole 单核 100% 空转，
此后所有控制台 API 调用永久阻塞：事件循环卡在 kbhit()、渲染线程卡在 WriteConsoleW，
命令早已跑完但结果永远没人收，Esc/Ctrl+C 全部失效，只能杀进程。

修法是结构性的：CREATE_NO_WINDOW 让子进程树得到**自己的**隐藏控制台，与 TUI 的
控制台物理隔离——不是修某一个触发条件，而是让「第二方动 TUI 的控制台」这件事
不可能发生。POSIX 没有共享控制台模式这回事，那边只需要把 stdin 掐掉。
"""
from __future__ import annotations

import sys

_CREATE_NO_WINDOW = 0x08000000


def isolated_console_kwargs() -> dict:
    """返回传给 subprocess / asyncio 子进程创建函数的隔离参数。

    Windows：creationflags=CREATE_NO_WINDOW（独立隐藏控制台）。
    其他平台：空 dict（creationflags 在 POSIX 上传非零值会抛 ValueError）。
    stdin 由调用方自己决定（DEVNULL 或 PIPE）——不要继承。
    """
    if sys.platform == "win32":
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}
