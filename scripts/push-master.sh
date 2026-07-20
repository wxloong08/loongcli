#!/usr/bin/env bash
# 推送 master 的确定性流程——替代人眼核对清单（闸门必须是代码，不赌细心）。
#
# 背景：master 与 dev 历史不相关（filter-repo 重写过），推送靠 read-tree 树对齐；
# dev 自 2026-07-19 起追踪内部文档（docs/tasks/CLAUDE.md/LOONG.md），read-tree
# 会把它们整棵带进 master 暂存区——含面试材料等隐私，绝不能上 GitHub。
# 本脚本：剔除（第一闸）+ 断言（第二闸），任一异常中止并回到 dev。
#
# 用法：bash scripts/push-master.sh "commit message"
set -euo pipefail

PRIVATE_PATHS=(docs tasks demo CLAUDE.md LOONG.md)
PRIVATE_RE='^(docs/|tasks/|demo/|CLAUDE\.md$|LOONG\.md$)'

msg="${1:?用法: bash scripts/push-master.sh \"commit message\"}"

branch=$(git branch --show-current)
if [ "$branch" != "dev" ]; then
    echo "⚠ 必须从 dev 出发（当前在 $branch）" >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "⚠ 工作区不干净，先提交或 stash" >&2
    exit 1
fi

git checkout master
# 从这里起，任何失败都要回到 dev，不把用户留在 master 的半成品状态上。
# 回程必须 -f：git rm --cached 后私有文件在 master 侧变成未跟踪文件，而 dev
# 跟踪它们，普通 checkout 会拒绝覆盖。-f 在此流程可证无损——磁盘内容就是
# read-tree 从 dev 铺出来的，且入口闸保证出发时 dev 工作区干净（无未提交编辑）。
trap 'git checkout -f dev >/dev/null 2>&1' ERR

git read-tree -u --reset dev

# 第一闸：确定性剔除私有路径
git rm -r -q --cached --ignore-unmatch "${PRIVATE_PATHS[@]}"

# 第二闸：断言暂存区零私有路径——剔除逻辑与私有集不同步时在此爆炸，而不是上网
if git diff --cached --name-only | grep -qE "$PRIVATE_RE"; then
    echo "⚠ 私有文件仍在暂存区，中止推送：" >&2
    git diff --cached --name-only | grep -E "$PRIVATE_RE" >&2
    git checkout -f dev
    exit 1
fi

if git diff --cached --quiet; then
    echo "master 已与 dev 对齐，无需推送"
    git checkout -f dev
    exit 0
fi

echo "── 待提交清单 ──"
git diff --cached --name-only
git commit -m "$msg"
git push origin master
git checkout -f dev
echo "✓ 已推送 master 并返回 dev"
