# 内网穿透 + 连接 GitCode上的NPU单卡资源

## Frp注册

**1. 注册 + 实名认证**（网页操作）

[SakuraFrp 用户中心](https://www.natfrp.com/user/)，记录访问密钥与管理密码。

密钥： 18dizr
密码： ww...

**2. 创建 TCP 隧道**（只做一次）

- 类型 `TCP`、本地地址 `127.0.0.1`、本地端口 `22`、选免费节点(我选：东莞双线PLUS2)
- 保存后得到隧道 ID `29946674`


## GitCode的notebook上安装和执行

```shell

sudo apt update && sudo apt install -y openssh-server && sudo apt install -y expect

echo 'root:123456' | sudo chpasswd
echo 'atomgit:123456' | sudo chpasswd

sudo sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo service ssh start

ACCESS_TOKEN="18d..."
WEBUI_PASSWORD="ww...hao"


# ACCESS_TOKEN="个人访问密钥"
# WEBUI_PASSWORD="管理密码"

export ACCESS_TOKEN WEBUI_PASSWORD

# 清理旧进程和旧文件，避免脚本中途询问 [y/N]
sudo pkill -f frpc
sudo pkill -f natfrp-service
sleep 1
sudo rm -f /home/natfrp/frpc /home/natfrp/natfrp-service
sudo rm -f /etc/natfrp/config.json

expect << 'EOF'
set timeout 300
set token $env(ACCESS_TOKEN)
set password $env(WEBUI_PASSWORD)

# 第一步：下载脚本到文件（不执行）
spawn bash -c "curl -sSL https://doc.natfrp.com/launcher.sh -o /tmp/launcher.sh"
expect eof

# 第二步：运行脚本文件，此时 stdin 是 pty
spawn sudo bash /tmp/launcher.sh

expect -re "访问密钥"
send -- "$token\r"

expect -re "远程管理密码"
send -- "$password\r"

expect -re "请再次输入"
send -- "$password\r"

expect -re "按 Enter 键以继续"
send -- "\r"

expect -re "输入回车键"
send -- "\r"

set timeout 10
expect eof
EOF

# sudo /home/natfrp/frpc -f 个人访问密钥:隧道id
sudo /home/natfrp/frpc -f 18..:29946674

```

## 本机powershell上连接GitCode的NoteBook

```shell
ssh-keygen -R "[frp-sea.com]:39188"  # 第1次这句可不执行
ssh -p 39188 atomgit@frp-sea.com
```


## 补充了解：内网穿透

内网穿透（NAT 穿透 / 端口映射隧道），简单说就是：让你的内网机器/服务，能被公网上的其他人访问到。

整个流程：

```text
外部用户 -> frp-sea.com:39188 -> 公网 frps -> 隧道 -> 内网 frpc -> 内网 SSH 服务
```

特点：
1. 内网机器主动向外连接，所以能穿过 NAT/防火墙。
2. 公网服务器只做中转/转发，真正服务还在内网。
3. 常用于：远程 SSH、远程桌面、访问家里 NAS、Web 服务、数据库、开发调试等



昨天： 

大EP分离
prefilling
LORA
动态shape子图优化   

昨天讨论的问题，了解下




