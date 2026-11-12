# 上传到 GitHub

压缩包已经按独立仓库整理。先解压，再上传目录中的文件，不要只把 ZIP 文件上传到仓库。

1. 登录 `zhang15250615625-ship-it`，新建公开仓库，名称为 `noise-conditioned-trajectory-diffusion`。
2. 创建时不额外生成 README 或许可证，使用包内已有 README 和第三方说明。
3. 在空仓库页面选择上传已有文件，将项目目录内的内容拖入并提交。`README.md`、`pyproject.toml`、`src/` 等应直接位于仓库根目录，不要再套一层同名文件夹。
4. `.gitignore` 是隐藏文件，也应一并上传。若文件管理器不显示隐藏文件，可使用下面的 Git 命令。

在解压后的项目根目录运行：

```bash
git init
git add .
git commit -m "Prepare portable diffusion research release"
git branch -M main
git remote add origin https://github.com/zhang15250615625-ship-it/noise-conditioned-trajectory-diffusion.git
git push -u origin main
```

Git 身份和 GitHub 登录使用你本机的已有配置；首次使用时按 Git 的提示配置。这里不包含或保存账号凭据。

上传成功后的项目链接为：

`https://github.com/zhang15250615625-ship-it/noise-conditioned-trajectory-diffusion`

该地址在你实际创建并上传之前只是预定地址，压缩包本身不会自动创建远程仓库。
