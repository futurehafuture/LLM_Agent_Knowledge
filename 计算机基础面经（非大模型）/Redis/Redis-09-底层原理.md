---
type: knowledge
status: active
created: 2026-06-12
updated: 2026-06-12
tags:
  - 面试/数据库
  - Redis
aliases:
  - Redis底层原理
  - Redis数据结构实现
  - Redis单线程
  - IO多路复用
---

# Redis 底层原理

## 1. 面试官到底想考什么

底层原理题能真正区分出候选人的深度。面试官想确认：

- 你是否真的理解 Redis 快的底层原因，而不只是背"纯内存"。
- 你能否解释 Redis 的核心数据结构是怎么实现的（跳表、哈希表、压缩列表等）。
- 你是否理解 IO 多路复用，以及 Redis 的事件模型。

## 2. SDS（简单动态字符串）

### 2.1 为什么不用 C 字符串

C 语言的字符串是 char 数组，以 `\0` 结尾。它有几个缺陷：

- 获取字符串长度需要 O(n) 遍历（strlen）。
- 追加字符串可能需要频繁重分配内存。
- 不是二进制安全（遇到 `\0` 就以为是字符串结尾）。

### 2.2 SDS 的结构

```c
struct sdshdr {
    int len;      // 字符串长度（O(1) 获取）
    int free;     // 剩余可用空间
    char buf[];   // 字符串内容（包含末尾 \0）
};
```

**优点：**

- **O(1) 获取长度**：直接读 len 字段。
- **预分配空间**：追加时不需要每次重分配（free 字段记录剩余空间）。
- **二进制安全**：用 len 判断结尾，不依赖 `\0`，可以存储任意字节。
- **防止缓冲区溢出**：追加前检查 free，不够就先扩容。

**空间预分配策略：**

- 字符串修改后小于 1MB：分配与 len 相同的空闲空间（double）。
- 字符串修改后大于等于 1MB：额外分配 1MB 空闲空间。

## 3. 字典（Hashtable）

### 3.1 哈希表结构

Redis 的字典由两个哈希表（ht[0] 和 ht[1]）组成，通常只用 ht[0]，ht[1] 用于渐进式 rehash。

```c
typedef struct dict {
    dictht ht[2];        // 两个哈希表
    long rehashidx;      // rehash 进度，-1 表示没有在 rehash
    // ...
} dict;

typedef struct dictht {
    dictEntry **table;   // 哈希表数组（桶）
    unsigned long size;  // 哈希表大小（2的幂）
    unsigned long sizemask;  // size - 1，用于取模
    unsigned long used;  // 已有节点数
} dictht;

typedef struct dictEntry {
    void *key;
    union { void *val; uint64_t u64; int64_t s64; } v;
    struct dictEntry *next;  // 链表（解决哈希冲突）
} dictEntry;
```

### 3.2 哈希冲突解决：链式哈希

当多个 key 的哈希值相同（落在同一个桶），用**链表**串起来：

```text
bucket[0] → entry_a → entry_b → NULL
bucket[1] → entry_c → NULL
bucket[2] → entry_d → entry_e → entry_f → NULL
```

新节点插入链表头部（O(1)）。

### 3.3 渐进式 Rehash

当负载因子（used / size）过高或过低时，需要扩容或缩容（rehash）。

Redis 不会一次性把所有数据迁移到新哈希表（那会阻塞很久），而是**渐进式 rehash**：

```text
1. 为 ht[1] 分配新的空间（ht[0] 的 2 倍）
2. 设置 rehashidx = 0（开始 rehash）
3. 每次对字典进行读写操作时，顺带把 ht[0] 中 rehashidx 对应的桶迁移到 ht[1]，然后 rehashidx++
4. rehash 期间，新写入的数据放到 ht[1]，查找在 ht[0] 和 ht[1] 都查
5. ht[0] 所有桶都迁移完成，将 ht[1] 设为 ht[0]，清空旧 ht[1]，rehashidx = -1
```

好处：把 rehash 的开销分摊到每次读写操作，避免一次性大规模阻塞。

## 4. 跳表（Skip List）

### 4.1 跳表是什么

跳表是 Sorted Set 的核心数据结构之一，也是面试高频考点。

普通有序链表查找是 O(n)，跳表通过**多层索引**实现 O(log n) 的查找。

```text
Level 3:  1 ←————————————————————→ 9
Level 2:  1 ←————→ 4 ←————————→ 9
Level 1:  1 ←→ 3 ←→ 4 ←→ 7 ←→ 9
Level 0:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
```

查找 7：

```text
Level 3：从 1 跳到 9，9 > 7，向下一层
Level 2：从 1 跳到 4，4 < 7，继续；从 4 跳到 9，9 > 7，向下一层
Level 1：从 4 跳到 7，找到！
```

### 4.2 跳表节点结构

```c
typedef struct zskiplistNode {
    sds ele;              // 元素（成员）
    double score;         // 分数
    struct zskiplistNode *backward;    // 后退指针（支持反向遍历）
    struct zskiplistLevel {
        struct zskiplistNode *forward;   // 该层的前进指针
        unsigned long span;              // 到下一个节点跳过了几个节点（用于计算排名）
    } level[];            // 层数组
} zskiplistNode;
```

### 4.3 跳表的性能

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 查找 | O(log N) | 多层索引加速 |
| 插入 | O(log N) | 找位置 + 更新索引 |
| 删除 | O(log N) | 找节点 + 更新索引 |
| 范围查询 | O(log N + M) | N 是定位成本，M 是结果数量 |
| 获取排名 | O(log N) | 利用 span 字段累加 |

### 4.4 为什么用跳表而不是红黑树

- **范围查询**：跳表找到最小值后顺序遍历即可，红黑树需要中序遍历，跳表更高效直观。
- **实现简单**：跳表代码比红黑树简单得多。
- **内存局部性**：跳表在范围遍历时内存访问模式更顺序，缓存更友好。
- **性能相近**：查找、插入、删除都是 O(log N)，和平衡树一样。

## 5. listpack（紧凑列表）

### 5.1 什么是 listpack

listpack（Redis 7.0+，之前叫 ziplist 压缩列表）是一种**内存连续的紧凑存储结构**，用于在元素数量较少时替代 hashtable、skiplist 等结构，节省内存。

```text
[total_bytes][last_entry_offset][num_entries][entry1][entry2]...[entryn][FF]
```

每个 entry 的格式：

```text
[previous_entry_length][encoding][content]
```

- `previous_entry_length`：前一个 entry 的长度，支持从后向前遍历。
- `encoding`：内容的编码类型（整数还是字节数组，以及长度）。
- `content`：实际数据。

### 5.2 为什么内存效率高

listpack 相比普通数组或链表更节省内存：

- 没有指针开销（链表每个节点有前后指针）。
- 可变长度编码（小整数用 1-2 字节，不需要固定 8 字节）。
- 内存连续，对 CPU 缓存友好。

**代价：**

- 插入删除需要移动后续元素，O(n) 复杂度。
- 更新一个元素可能引发"连锁更新"（previous_entry_length 字段大小改变）。

所以 listpack 只在数据量小时使用，超过阈值就切换到更高效的结构（hashtable、skiplist 等）。

## 6. intset（整数集合）

当 Set 里的元素都是整数且数量较少时，使用 intset：

```c
typedef struct intset {
    uint32_t encoding;    // 编码方式（INT16/INT32/INT64）
    uint32_t length;      // 元素个数
    int8_t contents[];    // 整数数组（升序排列）
} intset;
```

intset 是有序整数数组，查找用二分查找（O(log N)）。当出现超出当前编码范围的整数（如 int16 里插入一个 int32 的值）时，会自动升级编码，但不会降级。

## 7. Redis 的单线程模型

### 7.1 Redis 的线程模型演进

**Redis 6.0 之前：** 完全单线程

- 文件事件处理（网络 IO + 命令执行）：单线程
- 后台任务（AOF 刷盘、RDB 生成等）：独立线程

**Redis 6.0+：** 网络 IO 多线程，命令执行仍然单线程

- 网络读写：多个 IO 线程（提高网络 IO 吞吐）
- 命令解析和执行：主线程（单线程）

所以说"Redis 是单线程的"是正确的，但要补充"命令执行是单线程的"。

### 7.2 单线程为什么没有并发问题

因为命令是顺序执行的，不存在多个命令同时修改同一个 key 的情况：

```text
命令队列：[cmd1, cmd2, cmd3, cmd4, ...]
主线程：  cmd1执行完 → cmd2执行完 → cmd3执行完 → ...
```

每条命令是串行的，所以不需要加锁，也不会有竞态条件。

### 7.3 单线程的影响

- **O(n) 命令要小心**：KEYS *、HGETALL（大哈希）、SMEMBERS（大集合）、LRANGE（大列表）等命令可能执行很慢，阻塞其他请求。
- **生产环境禁用 KEYS ***：用 SCAN 替代（分批迭代，不阻塞）。

```bash
# 禁止在生产环境用
KEYS *               # 会阻塞主线程！

# 推荐用 SCAN 迭代
SCAN 0 MATCH "user:*" COUNT 100
# → cursor 和 结果列表，cursor=0 时迭代完成
```

## 8. IO 多路复用

### 8.1 为什么需要 IO 多路复用

Redis 要同时服务大量客户端连接（可能数万个）。

**传统做法：** 一个线程处理一个连接 → 需要数万个线程 → 线程切换开销大，内存消耗巨大。

**IO 多路复用：** 一个线程同时监听多个文件描述符（socket），哪个 socket 有数据来了就处理哪个。

### 8.2 三种多路复用机制

| 机制 | 特点 | 适用系统 |
|------|------|---------|
| select | 最古老，最多 1024 个 fd，每次调用需要复制 fd_set | Linux/Unix/Windows |
| poll | 取消了 1024 限制，但仍然是 O(n) 扫描 | Linux/Unix |
| epoll | O(1) 就绪通知，支持大量 fd，Linux 特有 | Linux |

Redis 在 Linux 上使用 epoll，是性能最优的选择。

### 8.3 epoll 工作原理

```text
1. epoll_create：创建 epoll 实例
2. epoll_ctl：注册要监听的 fd 和事件（读/写）
3. epoll_wait：等待，有就绪事件时返回就绪列表
4. 处理就绪 fd 上的事件
5. 循环
```

epoll 的关键优势：

- **边缘触发/水平触发**：可以精确控制事件通知时机。
- **就绪列表**：只返回有事件的 fd，不需要遍历所有 fd。
- **内核驱动**：fd 就绪时内核主动通知，O(1) 时间复杂度。

### 8.4 Redis 的文件事件模型

Redis 基于 Reactor 模式构建了一个事件处理器：

```text
IO 多路复用程序（epoll）
   ↓ 就绪事件列表
文件事件分发器
   ↓ 按事件类型分发
事件处理器：
  - 连接应答处理器（accept 新客户端）
  - 命令请求处理器（读取客户端命令）
  - 命令回复处理器（发送响应给客户端）
```

整个过程：

```text
客户端连接 → epoll 通知 → 连接应答处理器 → 建立连接
客户端发命令 → epoll 通知 → 命令请求处理器 → 解析 → 执行 → 放入队列
→ 命令回复处理器 → 把结果写回客户端
```

## 9. 总结：Redis 为什么快（完整版）

```text
1. 纯内存操作：所有数据在 RAM，不涉及磁盘 IO

2. 高效数据结构：
   - SDS 支持 O(1) 获取长度
   - listpack 内存连续，缓存友好
   - 跳表支持 O(log N) 的有序操作
   - 渐进式 rehash 避免一次性阻塞

3. 单线程命令执行：
   - 无锁竞争，无上下文切换
   - 命令串行执行，不需要同步原语

4. IO 多路复用：
   - epoll 实现，一个线程处理万级并发连接
   - O(1) 时间复杂度处理就绪事件

5. 网络协议精简（RESP 协议）：
   - 文本协议，解析简单
   - 请求响应模型，无额外协议开销
```

## 10. 面试回答模板

Redis 的底层原理可以从数据结构和事件模型两个维度来说。

数据结构层面：String 使用 SDS（支持 O(1) 获取长度，预分配空间）；Hash 和 Set 在元素少时用 listpack（内存连续），多时用 hashtable（链式哈希 + 渐进式 rehash）；Sorted Set 在元素多时用跳表+哈希表，跳表通过多层索引实现 O(log N) 的范围查询，比红黑树更适合范围查询场景。

事件模型层面：Redis 命令执行是单线程，不需要加锁，性能稳定可预测。底层用 IO 多路复用（Linux 上是 epoll），一个线程监听大量 socket 连接，哪个 socket 就绪了就处理哪个，实现了高并发连接的低开销处理。

## 我应该背下来的最短答案

Redis 底层：String 用 SDS（O(1) 获取长度）；Hash/Set 少用 listpack 多用 hashtable（渐进式 rehash）；Sorted Set 多时用跳表（O(log N) 范围查询，比红黑树易实现且适合范围查询）。事件模型：单线程命令执行（无锁）+ epoll IO 多路复用（高并发连接）= Redis 极速的底层原因。
