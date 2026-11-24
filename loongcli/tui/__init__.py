from loongcli.tui import cjk_wrap

# 中日韩文字之间也能断行（Rich 默认只在空格处断，中文段落会提前断出半截行），见 cjk_wrap.py
cjk_wrap.install()
