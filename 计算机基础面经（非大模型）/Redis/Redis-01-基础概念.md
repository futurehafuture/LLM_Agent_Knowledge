---
type: knowledge
status: active
created: 2026-06-12
updated: 2026-06-12
tags:
  - 面试/数据库
  - Redis
aliases:
  - Redis基础概念
  - Redis是什么
---

# Redis 基础概念

## 1. 面试官到底想考什么

问"Redis 是什么"或"Redis 为什么这么快"，面试官想考察的是：

- 你是否真正理解 Redis 和普通数据库、传统缓存的区别。
- 你是否能解释 Redis 高性能的根本原因，而不只是背"纯内存"三个字。
- 你能否把 Redis 的适用场景讲清楚，而不是说"Redis 缓存啥都行"。

## 2. 零基础版解释

想象你在图书馆查书。

- **MySQL** 就像图书馆的书库：书很多，分类很全，查询很准，但你每次都要走到书架前，找书，翻书，速度慢。
- **Redis** 就像你桌上的便利贴：最近常用的信息都贴在上面，拿起来就能看，速度极快，但桌面空间有限，放不了太多东西。

Redis 的本质：**把数据放在内存里，用起来非常快，但内存断电数据会消失，所以需要搭配持久化机制和后端存储**。

## 3. Redis 的正式定义

Redis 是一个开源的、**基于内存**（in-memory）的键值（key-value）存储系统，同时支持多种数据结构，并提供可选的持久化机制。

全称：Remote Dictionary Server（远程字典服务）。

版本历程：

- 2009 年：Salvatore Sanfilippo 发布第一版。
- 2015 年：Redis 3.0 发布，原生支持集群（Cluster）。
- 2020 年：Redis 6.0 发布，引入多线程 I/O 处理。
- 2022 年：Redis 7.0 发布，新增多项功能增强。

## 4. Redis 为什么这么快

这是最常被追问的问题。答案不能只说"内存"，要能解释清楚每一个原因。

### 4.1 纯内存操作

Redis 的数据存储在 RAM（内存）中，读写速度比磁盘快 10 万倍以上。

```text
内存读取速度：约 100 纳秒级别
磁盘随机读取：约 10 毫秒级别（HDD），约 0.1 毫秒级别（SSD）
差距：内存是磁盘随机读写的 1000 倍以上
```

MySQL 查询涉及磁盘 IO（哪怕有 Buffer Pool 缓存也会有 miss），Redis 全部在内存，几乎没有 IO 等待。

### 4.2 单线程模型（命令处理）

Redis 处理命令使用单线程，避免了多线程的锁竞争、上下文切换和同步开销。

很多人直觉上觉得"单线程不是更慢吗"，但这里有个关键认识：

**Redis 的性能瓶颈不在 CPU，而在内存和网络 IO。** 命令处理本身很快，CPU 几乎不是瓶颈。单线程反而让代码简单，不需要加锁，避免了多线程的各种竞态问题。

注意：Redis 6.0+ 引入了多线程 I/O（处理网络读写），但命令执行本身仍然是单线程的。

### 4.3 I/O 多路复用

Redis 使用 I/O 多路复用技术（epoll/select/kqueue）来处理大量并发连接。

```text
传统方式：一个线程处理一个连接，1000 个连接需要 1000 个线程
多路复用：一个线程同时监听 1000 个连接的就绪状态，有事件才处理
```

I/O 多路复用让单个线程能高效处理大量并发客户端，不需要为每个连接开一个线程。

### 4.4 高效的底层数据结构

Redis 对不同数据类型根据数据量选择不同的底层实现：

- 小 list 用 listpack（紧凑存储），大 list 用双向链表。
- 小 hash 用 listpack，大 hash 用 hashtable。
- Sorted Set 用 skiplist（跳表），支持 O(log N) 的快速查找。

这些数据结构针对内存使用和速度做了深度优化，比通用数据库的存储结构更轻量。

### 总结：Redis 快的四个原因

```text
1. 纯内存操作 - 避免磁盘 IO 延迟
2. 单线程命令处理 - 避免锁竞争和上下文切换
3. IO 多路复用 - 单线程处理大量并发连接
4. 优化的底层数据结构 - 内存紧凑，操作高效
```

## 5. Redis 的适用场景

### 5.1 缓存（最核心的使用场景）

把 MySQL 查询结果缓存到 Redis，下次请求直接从 Redis 取，避免打 MySQL。

```python
def get_user(user_id):
    # 先查 Redis
    user = redis.get(f"user:{user_id}")
    if user:
        return user
    # Redis 没有，查 MySQL
    user = mysql.query("SELECT * FROM user WHERE id = %s", user_id)
    # 写入 Redis，设置过期时间
    redis.setex(f"user:{user_id}", 3600, user)
    return user
```

### 5.2 计数器

Redis 的 INCR 命令是原子操作，天然适合做计数器。

```bash
INCR page_view:article:1001    # 文章点击数 +1
INCR user:1001:login_count     # 用户登录次数 +1
```

### 5.3 会话存储（Session）

Web 服务器把用户 Session 存到 Redis，多台服务器都能读取，实现无状态扩展。

```text
用户登录 → 生成 token → Redis.set("session:token", user_data, ex=1800)
用户请求 → 携带 token → Redis.get("session:token") → 验证身份
```

### 5.4 排行榜

Sorted Set 天然支持按分数排序，适合排行榜。

```bash
ZADD leaderboard 9800 "alice"
ZADD leaderboard 9500 "bob"
ZREVRANGE leaderboard 0 9 WITHSCORES   # 取前10名
```

### 5.5 消息队列（轻量级）

List 的 LPUSH/RPOP 或 BRPOP 可以模拟简单队列。Stream 类型支持更复杂的消息队列语义。

```bash
LPUSH task_queue "task1"         # 生产者入队
BRPOP task_queue 0               # 消费者阻塞等待出队
```

### 5.6 分布式锁

用 SET NX PX 命令实现分布式锁，多台服务器竞争同一把锁。

```bash
SET lock:resource_id "unique_value" NX PX 30000  # 加锁，30s 自动过期
```

### 5.7 限流

用 INCR + EXPIRE 实现接口限流。

```python
def rate_limit(user_id, max_requests=100, window=60):
    key = f"rate:{user_id}:{int(time.time() / window)}"
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window)
    return count <= max_requests
```

### 5.8 位图统计

Bitmap 适合用户签到、在线状态等大规模布尔值存储。

```bash
SETBIT user:signin:2026-06-12 1001 1   # 用户 1001 今天签到
GETBIT user:signin:2026-06-12 1001     # 查询签到状态
BITCOUNT user:signin:2026-06-12        # 当天签到总人数
```

## 6. Redis 和其他缓存方案的对比

| 特性 | Redis | Memcached | 本地缓存（HashMap） |
|------|-------|-----------|-----------------|
| 数据结构 | 多种（String/List/Hash等） | 只有 String | 任意 Java 对象 |
| 持久化 | 支持 | 不支持 | 不支持 |
| 集群 | 原生支持 | 客户端分片 | 不支持分布式 |
| 原子操作 | 支持 | 有限支持 | 需要加锁 |
| 发布订阅 | 支持 | 不支持 | 不支持 |
| 单机性能 | 约 10万QPS | 约 10万QPS | 更高（无网络） |
| 适用场景 | 缓存/队列/锁等多场景 | 简单缓存 | 单机缓存 |

## 7. Redis 的局限性

- **内存有限**：数据必须放在内存，大数据量成本高。
- **数据安全**：虽然有持久化，但与关系型数据库相比，数据持久性保证更弱。
- **不支持复杂查询**：不能像 SQL 一样做 JOIN、GROUP BY 等复杂查询。
- **单机性能有上限**：单线程处理命令，极高 QPS 下需要集群分担。

## 8. 面试回答模板

Redis 是基于内存的键值存储系统，支持 String、List、Hash、Set、Sorted Set 等多种数据结构，并提供可选的持久化机制。它之所以快，主要是因为：一、数据都在内存里，不涉及磁盘 IO；二、命令处理用单线程，没有锁竞争和上下文切换；三、用 IO 多路复用处理并发连接；四、底层数据结构针对内存做了优化。

Redis 最核心的用途是缓存，把高频读操作的结果缓存起来，降低数据库压力。此外还可以做计数器、会话存储、排行榜、分布式锁、限流等。

## 9. 追问方向

- Redis 单线程处理命令，那它如何支持高并发？
- Redis 6.0 引入多线程了吗？和之前的单线程有什么区别？
- Redis 持久化怎么做？（见 [[Redis-03-持久化|持久化]]）
- Redis 缓存和 MySQL 数据不一致怎么办？（见 [[Redis-07-缓存设计与常见问题|缓存设计]]）

## 10. 常见误区

- **误区 1：Redis 是多线程的。** 命令处理是单线程的，Redis 6.0 引入的多线程只是用于 I/O 读写，不是命令执行。
- **误区 2：Redis 数据不会丢。** 默认配置下 Redis 有丢失数据的风险，需要配置 AOF 等持久化方案。
- **误区 3：Redis 可以完全替代 MySQL。** 两者定位不同，Redis 做缓存和辅助，MySQL 做持久化存储。

## 我应该背下来的最短答案

Redis 是基于内存的键值存储系统，速度快的核心原因是：纯内存操作、单线程命令处理（无锁竞争）、IO 多路复用（处理大量并发连接）、优化的底层数据结构。主要用途是缓存、计数器、会话、排行榜、分布式锁等。
