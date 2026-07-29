# 参赛队代码目录

在本目录下创建 `teams/<队名>/`，实现 `TeamController` 并在官方基线上扩展优化。

```bash
cp -r teams/template teams/my_team
python scripts/evaluate.py --controller teams/my_team/controller.py
```

赛题目标：将 10 个箱子搬运至放货区并完成放置。详见 [赛题说明.md](../赛题说明.md)。
