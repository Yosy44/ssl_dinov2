- DrMaekawa_colonへマウント
sudo mount -t cifs //10.1.1.46/GAI/DrMaekawa_colon ~/mnt/DrMaekawa_colon \
  -o username=yyama,uid=$(id -u),gid=$(id -g),iocharset=utf8,vers=3.0

smb://10.1.1.46/GAI

- クローンのコマンド
git clone https://github.com/Yosy44/ssl_dinov2.git

- 確認のコマンド
cd ssl_dinov2
git status

[On branch main
nothing to commit, working tree clean]
がでればOK