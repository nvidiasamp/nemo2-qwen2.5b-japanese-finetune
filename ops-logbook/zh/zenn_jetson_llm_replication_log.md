# Zenn Jetson LLM 复现日志

## 日期: 2025年05月09日

### 博客步骤/当前目标:

- 分析 Zenn 博客文章《Jetson Orin Nano で日本語LLMを動かしてみた》([https://zenn.dev/headwaters/articles/jetson-orin-nano-llm](https://zenn.dev/headwaters/articles/jetson-orin-nano-llm)) 的 "1. SSD に Jetson OS と Jetson SDK をインストール" 步骤，并决定是否需要执行。

### 执行操作:

- 对比博客描述与当前 Jetson Orin Nano 的环境信息 (`jetson_environment_info.md`)。

### 观察记录/遇到的问题:

- **博客中步骤 "1. SSD に Jetson OS と Jetson SDK をインストール" 的核心内容包括：**
    1.  物理准备：安装 NVMe SSD，（可选）使用 `nvme-cli` 初始化 SSD，设置 Jetson 进入强制恢复模式。
    2.  宿主PC准备：需要 Ubuntu 20.04 PC（因为文章目标是安装 JetPack 5.1.3）。
    3.  使用 NVIDIA SDK Manager：在宿主 PC 上通过 SDK Manager 选择 JetPack 5.1.3 并将其刷写到 Jetson Orin Nano 的 NVMe SSD。
    4.  Jetson 初始设置：完成 OS 的首次启动配置。
- **当前 Jetson Orin Nano 环境信息摘要：**
    - 操作系统已安装在 NVMe SSD (`/dev/nvme0n1p1` 挂载为 `/`)。
    - 当前运行的是 JetPack 6.0 系列 (R36, Rev 2.0) 和 Ubuntu 22.04.5 LTS。
    - 无需进行 `nvme format` 底层格式化，因为系统已稳定运行。

### 解决方案/临时对策 (决定跳过步骤1的原因):

- **主要原因：** 当前系统已经满足了 Zenn 文章第一步操作的最终目的——即在 NVMe SSD 上拥有一个可运行的 JetPack 系统。
- **具体分析：**
    1.  **OS 已在 NVMe SSD 上：** 您的系统已经从 NVMe SSD 启动并运行，无需重新进行物理安装或使用 SDK Manager 从头刷写基础系统到 SSD 的操作。
    2.  **JetPack 版本差异：** Zenn 文章的目标是安装 JetPack 5.1.3 (需要 Ubuntu 20.04 宿主机)。您当前运行的是更新的 JetPack 6.0 系列。除非有特殊需求要降级到 JetPack 5.1.3 (这将是一个完整的、涉及宿主PC和恢复模式的刷机过程)，否则没有必要执行文章中的刷机步骤。
    3.  **初始化 SSD 非必需：** 文章中提到的 `nvme-cli format` 是针对裸盘或有问题磁盘在安装全新操作系统前的操作。既然您的系统已在 NVMe SSD 上稳定运行，此操作不仅非必需，而且会擦除现有系统。

### 心得/学到的知识:

- 确认了当前 Jetson Orin Nano 的系统状态已满足 Zenn 文章第一部分的核心要求。
- 理解了 Zenn 文章第一步主要是一个完整的系统刷写流程，适用于首次配置或特定版本安装。
- 决定跳过 Zenn 文章的第一步，直接从第二步 "2. Jetson Orin Nano の RAM の最適化" 开始进行环境优化和后续操作。

---

## 日期: 2025年05月09日

### 博客步骤/当前目标:

- 根据当前 Jetson Orin Nano (JetPack 6.0, Ubuntu 22.04, NVMe SSD 为根目录) 环境，制定并记录 Zenn 文章第二步 "2. Jetson Orin Nano の RAM の最適化" 的具体执行计划和注意事项。

### 针对当前环境的RAM优化执行计划:

**前提:** 目标是为 LLM 运行释放最大可用 RAM，并将当前 3.7Gi 的 `zram` 替换为约 16GB 基于 NVMe SSD 的 swap 文件。

#### 步骤 2.1: 禁用图形用户界面 (GUI)

- **操作规划:** 执行以下命令将系统默认启动目标更改为多用户命令行界面。
  ```bash
  sudo systemctl set-default multi-user.target
  ```
- **预期影响:** 重启后系统将进入命令行模式，不再加载桌面环境，预计释放约 800MB RAM。
- **如何恢复:** 如果之后想恢复桌面环境，可以使用 `sudo systemctl set-default graphical.target` 并重启。
- **记录 (2025年05月09日):**
    - **已执行命令:**
      ```bash
      dtc@ubuntu:~/wrokspace/LLM$ sudo systemctl set-default multi-user.target
      Removed /etc/systemd/system/default.target.
      Created symlink /etc/systemd/system/default.target → /lib/systemd/system/multi-user.target.
      ```
    - 此更改将在下次重启后生效。

#### 步骤 2.2: 禁用 nvargus-daemon 服务

- **操作规划:** 如果不使用摄像头功能，执行以下命令禁用相机服务。
  ```bash
  sudo systemctl disable nvargus-daemon.service
  ```
- **预期影响:** 释放少量系统资源。
- **如何恢复:** 如果之后需要使用相机服务，可以使用 `sudo systemctl enable nvargus-daemon.service` 并重启 (或 `sudo systemctl start nvargus-daemon.service` 立即启动)。
- **记录 (2025年05月09日):**
    - **已执行命令:**
      ```bash
      dtc@ubuntu:~/wrokspace/LLM$ sudo systemctl disable nvargus-daemon.service
      Removed /etc/systemd/system/multi-user.target.wants/nvargus-daemon.service.
      ```
    - 此更改将在下次重启后生效。

#### 步骤 2.3: Swap 空间配置 (核心调整部分)

**a. 调查并尝试禁用当前 `zram` 配置:**

- **操作规划1 (调查 `nvzramconfig.service`):**
  ```bash
  systemctl status nvzramconfig.service
  ```
- **操作规划2 (调查活动 zram 设备):**
  ```bash
  swapon -s
  lsblk | grep zram
  ```
- **记录 (2025年05月09日 - zRAM 调查结果):**
    - **执行 `systemctl status nvzramconfig.service` 输出:**
      ```
      ○ nvzramconfig.service - ZRAM configuration
           Loaded: loaded (/etc/systemd/system/nvzramconfig.service; enabled; vendor preset: enabled)
           Active: inactive (dead) since Thu 1970-01-01 09:00:59 JST; 55 years 4 months ago
         Main PID: 951 (code=exited, status=0/SUCCESS)
              CPU: 97ms

       1月 01 09:00:58 ubuntu nvzramconfig.sh[1122]: ラベルはありません, UUID=48746f39-f11b-4e67-a258-8dffff68c45f
       1月 01 09:00:58 ubuntu nvzramconfig.sh[1131]: スワップ空間バージョン 1 を設定します。サイズ = 635 MiB (665870336 バイト)
       1月 01 09:00:58 ubuntu nvzramconfig.sh[1131]: ラベルはありません, UUID=62ddb39b-3daa-49ec-8fa3-77745b4bab6c
       1月 01 09:00:58 ubuntu nvzramconfig.sh[1143]: スワップ空間バージョン 1 を設定します。サイズ = 635 MiB (665870336 バイト)
       1月 01 09:00:58 ubuntu nvzramconfig.sh[1143]: ラベルはありません, UUID=29f6745f-c1cc-4e19-a5b1-1dbc9aed4eed
       1月 01 09:00:58 ubuntu nvzramconfig.sh[1160]: スワップ空間バージョン 1 を設定します。サイズ = 635 MiB (665870336 バイト)
       1月 01 09:00:58 ubuntu nvzramconfig.sh[1160]: ラベルはありません, UUID=8039c050-414c-4579-9efa-32316dcf8330
      ```
    - **执行 `swapon -s` 及 `lsblk | grep zram` (综合输出观察):**
      ```
      Filename                                Type            Size            Used            Priority
      /dev/zram0                              partition       650264          0               5
      /dev/zram1                              partition       650264          0               5
      /dev/zram2                              partition       650264          0               5
      /dev/zram3                              partition       650264          0               5
      /dev/zram4                              partition       650264          0               5
      /dev/zram5                              partition       650264          0               5

      zram0        252:0    0   635M  0 disk [SWAP]
      zram1        252:1    0   635M  0 disk [SWAP]
      zram2        252:2    0   635M  0 disk [SWAP]
      zram3        252:3    0   635M  0 disk [SWAP]
      zram4        252:4    0   635M  0 disk [SWAP]
      zram5        252:5    0   635M  0 disk [SWAP]
      ```
    - **分析:**
        - `nvzramconfig.service` 服务是 `enabled` 状态，并在系统启动早期配置了6个约635MB的zram设备作为swap。
        - **结论：** `nvzramconfig.service` 是当前系统上负责创建和配置这些 `zram` Swap的服务。

- **操作规划3 (禁用 `nvzramconfig` 并临时移除 zRAM):**
  ```bash
  sudo systemctl stop nvzramconfig.service
  sudo systemctl disable nvzramconfig.service
  # 临时移除zram设备进行验证 (用户选择选项A)
  sudo swapoff /dev/zram0
  sudo swapoff /dev/zram1
  sudo swapoff /dev/zram2
  sudo swapoff /dev/zram3
  sudo swapoff /dev/zram4
  sudo swapoff /dev/zram5
  # 验证命令
  swapon -s
  lsblk | grep zram
  free -h
  ```
- **预期影响:** 停止并禁止现有 zram swap 的自动加载，并临时移除当前活动的zram swap。
- **记录 (2025年05月09日 - 执行禁用与验证zRAM):**
    - **执行 `sudo systemctl stop nvzramconfig.service` 和 `sudo systemctl disable nvzramconfig.service`:** (用户报告无错误输出)
    - **执行 `sudo swapoff /dev/zramX` (X从0到5):** (用户报告无错误)
    - **验证命令输出:**
      ```bash
      # swapon -s (从free -h推断为空)

      # lsblk | grep zram
      zram0        252:0    0   635M  0 disk 
      zram1        252:1    0   635M  0 disk 
      zram2        252:2    0   635M  0 disk 
      zram3        252:3    0   635M  0 disk 
      zram4        252:4    0   635M  0 disk 
      zram5        252:5    0   635M  0 disk 

      # free -h
                         total        used        free      shared  buff/cache   available
      Mem:           7.4Gi       2.3Gi       2.7Gi        38Mi       2.5Gi       4.9Gi
      Swap:             0B          0B          0B
      ```
    - **执行结果分析:**
        - `nvzramconfig.service` 已成功停止和禁用。
        - 所有 `zram` 设备已成功通过 `swapoff` 从活动 Swap 中移除。
        - `lsblk` 输出显示 `zram` 块设备节点依然存在，但不再作为 `[SWAP]` 使用。
        - `free -h` 输出明确显示当前系统 Swap 为 0B。
        - **结论: 成功清除了现有的 zram Swap，可以继续创建新的基于 SSD 的 Swap 文件。**

**b. 创建新的 Swap 文件 (在 NVMe SSD 上):**

- **决策:** 确定 Swap 文件路径为 `/16GB_swapfile`。
- **操作规划1 (创建文件):**
  ```bash
  sudo fallocate -l 16G /16GB_swapfile
  ```
- **操作规划2 (设置权限):**
  ```bash
  sudo chmod 600 /16GB_swapfile
  ```
- **记录 (2025年05月09日 - 创建Swap文件):**
    - **已执行命令:**
      ```bash
      dtc@ubuntu:~/workspace$ sudo fallocate -l 16G /16GB_swapfile
      [sudo] dtc のパスワード: 
      dtc@ubuntu:~/workspace$ sudo chmod 600 /16GB_swapfile
      ```
    - **分析:** `fallocate` 命令成功执行 (在输入正确密码后)，预分配了16GB空间。`chmod 600` 命令成功设置了文件权限，确保只有root用户可读写。

**c. 格式化并激活 Swap 文件:**

- **操作规划1 (格式化):**
  ```bash
  sudo mkswap /16GB_swapfile
  ```
- **操作规划2 (激活):**
  ```bash
  sudo swapon /16GB_swapfile
  ```
- **记录 (2025年05月09日 - 格式化并激活Swap):**
    - **已执行命令及输出:**
      ```bash
      dtc@ubuntu:~/workspace$ sudo mkswap /16GB_swapfile
      スワップ空間バージョン 1 を設定します。サイズ = 16 GiB (17179865088 バイト)
      ラベルはありません, UUID=75e6a628-abd9-4002-937f-40cd6c6368ab
      dtc@ubuntu:~/workspace$ sudo swapon /16GB_swapfile
      ```
    - **分析:** `mkswap` 命令成功将文件格式化为Swap空间，并分配了UUID。`swapon` 命令成功激活了该Swap文件。此时可以使用 `swapon -s` 或 `free -h` 验证。

**d. 使 Swap 文件永久生效 (修改 `/etc/fstab`):**

- **操作规划 (安全方法):**
  ```bash
  echo '/16GB_swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```
- **验证规划:** `cat /etc/fstab`
- **记录 (2025年05月09日 - 修改fstab并处理重复条目):**
    - **已执行命令 (添加条目):**
      ```bash
      dtc@ubuntu:~/workspace$ echo '/16GB_swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
      [sudo] dtc のパスワード: 
      /16GB_swapfile none swap sw 0 0
      ```
    - **已执行命令 (验证fstab):**
      ```bash
      dtc@ubuntu:~/workspace$ cat /etc/fstab
      # /etc/fstab: static file system information.
      #
      # These are the filesystems that are always mounted on boot, you can
      # override any of these by copying the appropriate line from this file into
      # /etc/fstab and tweaking it as you see fit.  See fstab(5).
      #
      # <file system> <mount point>             <type>          <options>                               <dump> <pass>
      /dev/root            /                     ext4           defaults                                     0 1
      /16GB_swapflie  none  swap  sw 0  0  # (注意: 此处有一个拼写错误 'swapflie'，实际添加的是 'swapfile')
      /16GB_swapfile none swap sw 0 0
      ```
    - **问题发现与处理:**
        - `cat /etc/fstab` 的输出显示 `/16GB_swapfile none swap sw 0 0` 条目被添加了两次。
        - (用户后续反馈) 用户已使用 `vi` 编辑器手动打开 `/etc/fstab` 并删除了其中一个重复的 `/16GB_swapfile none swap sw 0 0` 条目，以确保配置文件的正确性和整洁性。同时，用户也修正了之前可能存在的 `/16GB_swapflie` 拼写错误（如果存在）。
        - **最终状态:** `/etc/fstab` 中现在应该只有一行正确的 `/16GB_swapfile none swap sw 0 0` 条目。

#### 步骤 2.4: 重启系统使初步优化生效

- **操作规划:**
  ```bash
  sudo reboot
  ```
- **记录 (2025年05月12日 - 重启后验证):**
    - **已执行命令:** `sudo reboot`
    - **重启后执行的验证命令及输出:**
        1.  `free -h`
            ```
            dtc@ubuntu:~/workspace$ free -h
                           total        used        free      shared  buff/cache   available
            Mem:           7.4Gi       806Mi       5.9Gi        34Mi       770Mi       6.4Gi
            Swap:           15Gi          0B        15Gi
            ```
        2.  `swapon -s`
            ```
            dtc@ubuntu:~/workspace$ swapon -s
            Filename                                Type            Size            Used            Priority
            /16GB_swapfile                          file            16777212        0               -2
            ```
        3.  `systemctl get-default`
            ```
            dtc@ubuntu:~/workspace$ systemctl get-default
            multi-user.target
            ```
        4.  `sudo systemctl status nvargus-daemon.service`
            ```
            dtc@ubuntu:~/workspace$ sudo systemctl status nvargus-daemon.service
            [sudo] dtc のパスワード: 
            ○ nvargus-daemon.service - Argus daemon
                 Loaded: loaded (/etc/systemd/system/nvargus-daemon.service; disabled; vendor preset: enabled)
                 Active: inactive (dead)
            
             5月 12 14:44:46 ubuntu systemd[1]: /etc/systemd/system/nvargus-daemon.service:42: Standard output type syslog >
            ```
    - **验证结果分析:**
        - **内存与Swap:** `free -h` 显示内存使用量从GUI模式下的约2.3Gi降低到约806Mi，可用内存显著增加。15Gi的Swap空间 (`/16GB_swapfile`) 已成功激活并被系统识别，当前未使用。
        - **活动Swap设备:** `swapon -s` 确认 `/16GB_swapfile` 是当前唯一活动的Swap设备，旧的zRAM已不存在。
        - **默认启动目标:** `systemctl get-default` 确认系统默认启动到 `multi-user.target` (命令行界面)。
        - **nvargus-daemon服务:** `systemctl status nvargus-daemon.service` 确认该服务处于 `disabled` 和 `inactive (dead)` 状态。
        - **结论:** 所有RAM优化相关的初步设置 (GUI禁用, nvargus禁用, 新Swap配置) 均已在重启后成功生效。

#### 步骤 2.5: [可选] 进一步禁用不必要的服务

**a. 列出当前启用的服务:**

- **操作规划:**
  ```bash
  systemctl list-unit-files --type=service --state=enabled
  ```
- **预期影响:** 输出当前系统所有开机自启的服务列表。
- **记录:** (将在实际操作后记录命令输出的服务列表，用于后续决策)

**b. 根据需求选择性禁用服务:**

- **参考 Zenn 文章列表，并结合自身需求判断。** 特别注意 `wpa_supplicant.service` (若使用Wi-Fi则保留) 和 `snapd.service` (若使用Snap包则保留或谨慎处理)。
- **示例 (仅当确认不需要时才执行):**
  ```bash
  # sudo systemctl disable bluetooth.service
  # sudo systemctl disable snap.cups.cupsd.service
  # ... (其他来自Zenn文章列表中的服务)
  ```
- **记录 (2025年05月12日 - 禁用可选服务):**
    - **背景：** 根据 Zenn 博客文章 ([https://zenn.dev/headwaters/articles/jetson-orin-nano-llm](https://zenn.dev/headwaters/articles/jetson-orin-nano-llm)) 的建议，并结合当前Jetson Orin Nano的使用场景（有线网络连接、使用SSH、NVIDIA核心服务以及可能使用Docker），为进一步优化系统资源以运行LLM，禁用了以下被认为非必要的服务。这些服务主要与蓝牙、无线网络、移动调制解调器、VPN以及Snap包管理相关，在当前专注运行LLM的场景下，禁用它们有助于释放系统资源。
    - **执行的命令 (根据用户提供的终端操作日志概览):**
      ```bash
      sudo systemctl disable bluetooth.service
      sudo systemctl disable ModemManager.service
      sudo systemctl disable wpa_supplicant.service
      sudo systemctl disable openvpn.service
      sudo systemctl disable snapd.apparmor.service
      sudo systemctl disable snapd.autoimport.service
      sudo systemctl disable snapd.core-fixup.service
      sudo systemctl disable snapd.recovery-chooser-trigger.service
      sudo systemctl disable snapd.seeded.service
      sudo systemctl disable snapd.service
      sudo systemctl disable snapd.system-shutdown.service
      ```
    - **详细说明与恢复方法：**
        - **`bluetooth.service`**
            - **禁用原因：** 系统不计划连接蓝牙设备（如键盘、鼠标等），禁用此服务可以节省资源。这与 Zenn 博文中的建议一致，旨在为LLM运行释放更多资源。
            - **如何恢复：** `sudo systemctl enable bluetooth.service`
        - **`ModemManager.service`**
            - **禁用原因：** 系统使用有线以太网连接，不需要管理移动宽带调制解調器（如3G/4G/5G网卡）。禁用此服务符合 Zenn 博文的优化建议。
            - **如何恢复：** `sudo systemctl enable ModemManager.service`
        - **`wpa_supplicant.service`**
            - **禁用原因：** 系统使用有线以太网连接，不需要 Wi-Fi Protected Access (WPA/WPA2/WPA3) 的客户端支持。禁用此服务符合 Zenn 博文的优化建议。
            - **如何恢复：** `sudo systemctl enable wpa_supplicant.service`
        - **`openvpn.service`**
            - **禁用原因：** 当前系统环境不计划使用 OpenVPN 进行虚拟专用网络连接。禁用此服务符合 Zenn 博文的优化建议。
            - **如何恢复：** `sudo systemctl enable openvpn.service`
        - **`snapd.*` 系列服务** (包括 `snapd.apparmor.service`, `snapd.autoimport.service`, `snapd.core-fixup.service`, `snapd.recovery-chooser-trigger.service`, `snapd.seeded.service`, `snapd.service`, `snapd.system-shutdown.service`)
            - **禁用原因：** 如果不依赖通过 Snap 包管理器安装的应用程序来运行LLM或其相关的开发工具，禁用 Snapd 相关的所有服务可以显著减少后台活动和资源占用。这与 Zenn 博文中的深度优化建议一致。
            - **如何恢复：** 如果将来需要使用 Snap 包，可以重新启用这些服务。通常，启用核心的 `snapd.service` 和 `snapd.socket` 可能就足够，其他服务会作为依赖被拉起。
              ```bash
              sudo systemctl enable snapd.service
              sudo systemctl enable snapd.socket 
              # 根据需要也可能要启用以下服务:
              # sudo systemctl enable snapd.apparmor.service
              # sudo systemctl enable snapd.autoimport.service
              # sudo systemctl enable snapd.core-fixup.service
              # sudo systemctl enable snapd.recovery-chooser-trigger.service
              # sudo systemctl enable snapd.seeded.service
              # sudo systemctl enable snapd.system-shutdown.service
              ```
              (恢复时，建议先启用 `snapd.service` 和 `snapd.socket`，然后根据具体错误或需求启用其他相关服务，并可能需要重启。)
    - **重要提示：** 在禁用这些服务后，强烈建议重启系统 (`sudo reboot`)，以确保所有更改完全生效，并仔细验证系统的核心功能（如网络连接、SSH远程访问、以及您计划使用的Docker服务等）是否仍然按预期工作。

**c. [可选] 再次重启系统:**

- **操作规划 (如果执行了步骤2.5.b):**
  ```bash
  sudo reboot
  ```
- **预期影响:** 使可选服务禁用生效。
- **记录 (2025年05月12日 - 执行重启):**
    - 用户已执行 `sudo reboot` 命令。

**d. 重启后系统状态验证及待处理问题 (2025年05月12日):**

- **背景:** 在执行一系列服务禁用操作并重启后，对系统核心功能和服务状态进行验证。

- **1. 核心功能验证结果:**
    - **内存与 Swap:**
        - `free -h` 显示：总内存约 7.4Gi，已用约 756Mi，可用约 6.5Gi。Swap 总量 15Gi，已用 0B。
        - `swapon -s` 显示：`/16GB_swapfile` 是唯一活动的 Swap 设备。
        - **结论:** 内存占用显著降低，Swap 配置正确，符合预期。
    - **网络连接 (有线):**
        - `ip a` 显示：`eth0` 接口 UP 并获得 IP 地址 (例如 `10.204.222.153/24`)。
        - `ping -c 3 google.com` 显示：成功通讯，0% 丢包。
        - **结论:** 有线网络连接正常，可访问外部网络。
    - **SSH 服务:**
        - `sudo systemctl status ssh` 显示：服务 `active (running)`。
        - **结论:** SSH 服务运行正常。

- **2. 已禁用服务状态确认 (部分成功):**
    - **成功禁用的服务 (状态为 `disabled` 且 `inactive (dead)`):**
        - `bluetooth.service`
        - `ModemManager.service`
        - `openvpn.service`
        - `snapd.apparmor.service`
        - `snapd.autoimport.service`
        - `snapd.core-fixup.service`
        - `snapd.recovery-chooser-trigger.service`
        - `snapd.seeded.service`
        - `snapd.service` (及其关联的 `snapd.socket`)
        - `snapd.system-shutdown.service`
    - **基本符合禁用预期的服务 (因条件不满足而未启动，状态 `enabled` 但 `inactive (dead)`):**
        - `ubuntu-advantage.service`
        - `ua-reboot-cmds.service`
        - `sssd.service`

- **3. 待处理的问题和观察:**
    - **a. Docker 服务启动失败:**
        - **症状:** `sudo systemctl status docker` 显示服务状态为 `failed (Result: exit-code)`。`docker ps` 命令因守护进程未运行而报错权限不足。
        - **相关日志 (`journalctl -p 3 -xb` 和 `journalctl -u docker.service`):** 系统日志多次记录 "Failed to start Docker Application Container Engine"。具体的 `docker.service` 日志需要进一步分析以确定根本原因 (例如，`sudo journalctl -u docker.service --since "10 minutes ago"` 的输出)。
        - **临时决定:** 用户当前不急需 Docker，此问题已记录，待将来需要使用 Docker 时再进行详细排查和修复。可能的排查方向包括检查 Docker 配置、依赖项、以及与其他系统服务（如 `containerd.service`）的交互。
    - **b. `wpa_supplicant.service` 状态异常:**
        - **症状 (2025-05-12, 首次检查):** `sudo systemctl status wpa_supplicant.service` 显示服务状态为 `Loaded: ... disabled ... Active: active (running)`。
        - **症状 (2025-05-12, 再次重启后检查):** 状态与首次检查基本一致，仍为 `Loaded: ... disabled ... Active: active (running)`。PID 为 890 (示例)。
        - **相关日志 (`journalctl -p 3 -xb`):** 记录了 `wpa_supplicant[PID]: nl80211: kernel reports: Registration to specific type not supported` 以及 NetworkManager 相关的 `device (wlan0): Couldn't initialize supplicant interface: Name owner lost`。
        - **分析:** 尽管服务被设置为 `disabled`，但似乎仍被 NetworkManager 或其他机制尝试激活，但由于内核接口问题未能完全正常工作。由于当前使用有线网络，此问题不影响核心功能。
        - **用户决定 (2025-05-12):** 用户表示有时可能会使用 Wi-Fi，因此决定暂时保留 `wpa_supplicant.service` 的当前状态，不进一步尝试停止或强制禁用它。尽管其状态为 `disabled` 但 `active (running)` 且日志中存在 `nl80211` 相关报错，但鉴于有线网络工作正常，此服务当前不影响主要操作。
        - **临时决定:** 已记录此现象和用户决定。此问题在再次重启后依然存在。将来若用户在尝试使用 Wi-Fi 时遇到问题，或希望彻底清理无线相关服务，应回顾此处的记录，并可能需要进一步排查 NetworkManager 配置或 `wpa_supplicant` 的行为。若不再需要 Wi-Fi 功能，届时可尝试 `sudo systemctl stop wpa_supplicant.service` 并观察效果。
    - **c. 部分建议禁用的服务状态仍为 `enabled` (状态更新于 2025-05-12 第二次重启后检查):**
        - **`nvweston.service`:**
            - **旧状态:** `Loaded: ... enabled ... Active: inactive (dead)`.
            - **新状态 (2025-05-12 第二次重启后):** `Loaded: ... disabled ... Active: inactive (dead)`.
            - **说明:** 服务已成功禁用。
        - **`avahi-daemon.service`:**
            - **旧状态:** `Loaded: ... enabled ... Active: active (running)`.
            - **新状态 (2025-05-12 第二次重启后):** `Loaded: ... disabled ... Active: inactive (dead)`.
            - **说明:** 服务已成功停止并禁用。
        - **`power-profiles-daemon.service`:**
            - **旧状态:** `Loaded: ... enabled ... Active: inactive (dead)`.
            - **新状态 (2025-05-12 第二次重启后):** `Loaded: ... disabled ... Active: inactive (dead)`.
            - **说明:** 服务已成功禁用。
        - **`switcheroo-control.service`:**
            - **旧状态:** `Loaded: ... enabled ... Active: inactive (dead)`.
            - **新状态 (2025-05-12 第二次重启后):** `Loaded: ... disabled ... Active: inactive (dead)`.
            - **说明:** 服务已成功禁用。
        - **总结:** 除 `wpa_supplicant.service` 外，之前列出的其他建议禁用的服务在本次检查中均已确认为 `disabled` 和 `inactive (dead)` 状态，表明系统服务优化进展良好。

### 接下来关注的重点:

- 用户已决定暂时不修复 Docker 问题，优先记录当前系统状态。
- 系统优化已取得显著进展，内存占用降低，Swap配置完成。
- 若后续需要使用 Docker，需回顾上述记录进行问题排查。
- 若需进一步精简服务，可处理 `wpa_supplicant.service` 和其他仍标记为 `enabled` 的服务。
- 继续 Zenn 博客文章的后续步骤，例如 "3. text-generation-webui で LLM をロードする"。

---

## 日期: 2025年05月12日 (续)

### 博客步骤/当前目标:

- 排查并修复之前遇到的 Docker 服务启动失败问题。
- 解决普通用户 `dtc` 无法执行 `docker` 命令的权限问题。

### Docker 服务故障排查与修复:

**1. 问题现象回顾:**
   - 在系统优化并重启后，尝试使用 `docker` 相关命令（如 `docker ps`）时，非 root 用户 `dtc` 遇到 `permission denied while trying to connect to the Docker daemon socket` 错误。
   - 使用 `sudo systemctl status docker` 检查服务状态，显示为 `inactive (dead)` 或 `failed`。

**2. 诊断过程 (执行命令及关键输出):**
   - `sudo systemctl restart docker` 后 `sudo systemctl status docker | cat`:
     - 服务状态显示为 `active (running)`，表明服务已成功启动。
   - `sudo journalctl -u docker.service -n 100 --no-pager`:
     - 日志显示之前的多次启动失败，根本原因为 iptables 规则添加失败:
       ```
       failed to start daemon: Error initializing network controller: ... unable to add return rule in DOCKER-ISOLATION-STAGE-1 chain:  (iptables failed: iptables v1.8.7 (nf_tables):  RULE_APPEND failed (No such file or directory): rule in chain DOCKER-ISOLATION-STAGE-1
       ```
     - 但日志末尾显示服务已成功启动。
   - `sudo systemctl status containerd | cat` 及 `sudo journalctl -u containerd.service -n 50 --no-pager`:
     - `containerd` 服务状态正常 (`active (running)`), 日志无明显错误。
   - `sudo cat /etc/docker/daemon.json`:
     - 配置文件内容正常，包含 NVIDIA runtime 配置。

**3. 修复结论:**
   - Docker 服务本身已恢复正常运行。之前的 iptables 问题似乎已自动解决或因其他操作间接修复。

### Docker 用户权限配置:

**1. 问题:**
   - 即便 Docker 服务已恢复运行，用户 `dtc` 仍无法直接执行 `docker` 命令（如 `docker ps`），报错 `permission denied while trying to connect to the Docker daemon socket`，需要 `sudo` 才能执行。

**2. 排查与尝试过程 (详细):**
   - **步骤 2.1: 将用户添加到 `docker` 组 (已执行)**
     ```bash
     sudo usermod -aG docker dtc
     ```
   - **步骤 2.2: 尝试在当前会话应用新组权限 (未生效)**
     ```bash
     newgrp docker
     ```
     - **结果:** 执行 `docker ps` 后，仍然出现 `permission denied` 错误。`newgrp` 会启动一个子 shell，但可能并未完全继承或应用到当前环境所需的所有变量或设置，或者对于 Docker Socket 的访问权限生效机制，仅靠 `newgrp` 不足。
   - **步骤 2.3: 完全退出 SSH 并重新登录 (未生效)**
     - **操作:** 用户断开 SSH 连接，然后重新登录。
     - **结果:** 执行 `docker ps` 后，依然是 `permission denied` 错误。这排除了 SSH 会话缓存旧权限的可能性，说明问题根源更深。
   - **步骤 2.4: 深入检查权限配置 (确认配置正确)**
     - **检查用户组成员 (`id dtc`):** 确认 `groups=...,994(docker)`，用户 `dtc` **确实** 在 `docker` 组 (GID 994) 中。
     - **检查组信息 (`getent group docker`):** 确认 `docker:x:994:dtc`，`docker` 组存在且 `dtc` 是成员。
     - **检查 Socket 权限 (`ls -l /var/run/docker.sock`):** 确认权限为 `srw-rw---- 1 root docker ...`，所有者是 `root`，组是 `docker`，并且组具有读写权限 (`rw`)。配置完全符合预期。
   - **步骤 2.5: 再次尝试强制应用权限和检查其他因素 (未解决)**
     - **确认 Docker 服务状态 (`sudo systemctl status docker | cat`):** 服务 `active (running)`。
     - **强制重新应用 Socket 权限 (`sudo chgrp docker /var/run/docker.sock && sudo chmod g+rw /var/run/docker.sock`):** 命令执行成功，但 `docker ps` 依然失败。
     - **检查 AppArmor (`sudo aa-status | cat`):** 显示 `apparmor not present.`，排除 AppArmor 干扰。
     - **尝试 `docker info`:** 同样报 `permission denied` 错误。

**3. 分析与结论:**
   - 所有常规的权限配置检查（用户组成员、Socket 文件权限）均显示**正确无误**。
   - 尝试的即时生效方法（`newgrp`）和标准生效方法（重新登录 SSH）都**未能解决**问题。
   - 强制应用权限和检查 AppArmor 也未发现异常。
   - 这种情况非常罕见，暗示可能存在更深层次的系统状态问题或某些服务间的潜在冲突，导致即使用户组权限设置正确，守护进程的 Socket 仍然无法被非 root 用户访问。

**4. 最终解决方案与验证:**
   - **操作:** 用户执行 `sudo reboot` **重启了整个 Jetson 系统**。
   - **结果:** 重启后，重新登录 SSH，用户 `dtc` 直接执行 `docker ps` 和 `docker info` 命令 **成功**，不再出现权限错误。
   - **最终结论:** 尽管具体原因不明，但**系统重启**彻底清除了可能存在的任何残留状态或冲突，完全应用了用户组的更改，最终解决了 Docker 的用户权限问题。Docker 环境现已对用户 `dtc` 完全可用。

### 下一步:

- Docker 服务和用户权限均已修复。
- 继续执行 Zenn 博客第三部分的 `jetson-containers` 相关步骤。

---

## 术语解释

### Jetson Orin Nano

- **是什么：** NVIDIA 公司推出的一款小型、低功耗、高性能的边缘计算设备（或称为模块）。它属于 Jetson Orin 系列的一员，专为在设备端直接运行人工智能 (AI) 和机器学习 (ML) 应用而设计。通常，您会购买一个包含 Jetson Orin Nano 模块和参考载板的开发者套件 (Developer Kit) 来进行开发。
- **用来干什么的：** 用于构建和部署 AI 驱动的嵌入式系统，例如智能机器人、自主无人机、智能摄像头、便携式医疗设备等。它可以在本地处理复杂的 AI 推理任务，而无需完全依赖云端计算。

### JetPack 6.0

- **是什么：** NVIDIA 为其 Jetson 系列边缘 AI 平台提供的综合性软件开发套件 (SDK)。JetPack 6.0 是指该 SDK 的一个特定版本。
- **用来干什么的：** 它包含了运行 Jetson 设备并开发 AI 应用所需的一切软件组件，主要包括：
    - **Jetson Linux (L4T - Linux for Tegra)：** 专门为 Jetson 优化的 Linux 操作系统。
    - **CUDA Toolkit：** NVIDIA 的并行计算平台和编程模型，允许开发者利用 GPU 进行通用计算。
    - **cuDNN：** NVIDIA CUDA 深度神经网络库，为深度学习框架提供 GPU 加速。
    - **TensorRT：** NVIDIA 的高性能深度学习推理优化器和运行时库，可以加速深度学习模型在 NVIDIA GPU 上的推理性能。
    - **OpenCV, VisionWorks 等视觉和图像处理库。**
    - **驱动程序和其他开发工具。**

    JetPack 简化了 Jetson 平台的软件安装和管理过程。

### SSD (Solid State Drive - 固态硬盘)

- **是什么：** 一种使用固态电子存储芯片阵列制成的硬盘。与传统的机械硬盘 (HDD) 不同，它没有移动部件。
- **用来干什么的：** 用作计算机的存储设备，用于安装操作系统、应用程序和存储用户数据。相比机械硬盘，SSD 具有更快的读写速度、更低的延迟、更好的抗震性、更低的噪音和功耗。在 Jetson 设备上使用 SSD (尤其是 NVMe SSD) 可以显著提升系统响应速度和数据加载速度。

### SDK (Software Development Kit - 软件开发工具包)

- **是什么：** 通常是一组软件开发工具的集合，允许开发者为特定的软件包、软件框架、硬件平台或操作系统创建应用程序。
- **用来干什么的：** SDK 提供了必要的库、API (应用程序编程接口)、代码示例、文档、调试工具等，帮助开发者更高效地构建、测试和部署软件。例如，JetPack 就是 Jetson 平台的 SDK。

### NVMe SSD (Non-Volatile Memory Express SSD - 非易失性内存主机控制器接口规范固态硬盘)

- **是什么：** 一种使用 NVMe 接口标准的高性能固态硬盘。NVMe 是一种专为 SSD 设计的通信接口和驱动程序，旨在充分利用 PCIe (Peripheral Component Interconnect Express) 总线的高带宽。
- **用来干什么的：** 与传统的 SATA 接口 SSD 相比，NVMe SSD 提供更高的吞吐量和更低的延迟，从而带来更快的启动速度、应用程序加载时间和文件传输速度。在 Jetson Orin Nano 这样的设备上使用 NVMe SSD，可以最大程度地发挥其处理性能，特别是在需要快速读写大量数据的 AI 应用中。

--- 