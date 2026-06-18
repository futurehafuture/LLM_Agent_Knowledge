---
type: knowledge
status: active
created: 2026-06-12
updated: 2026-06-12
tags:
  - 面试/数据库
  - Redis
aliases:
  - 缓存穿透
  - 缓存击穿
  - 缓存雪崩
  - Redis缓存一致性
---

# Redis 缓存设计与常见问题

## 1. 面试官到底想考什么

缓存三大问题（穿透、击穿、雪崩）是 Redis 面试里最高频的题目，几乎 100% 会问。

面试官想确认：

- 你能否准确区分三个概念（很多人搞混）。
- 你有没有真正思考过每种场景的解决方案，而不只是背单词。
- 你对缓存一致性问题是否有认识。

## 2. 缓存穿透（Cache Penetration）

### 2.1 什么是缓存穿透

**定义：** 查询一个**根本不存在**的数据，缓存里没有，数据库里也没有，每次请求都直接打到数据库。

```text
正常请求：
请求 → Redis（命中）→ 返回

缓存穿透：
请求(id=-1，不存在) → Redis（不命中）→ MySQL（也没有）→ 返回空
下次同样请求 → Redis（不命中）→ MySQL（也没有）→ ...
每次都打到 MySQL，形成穿透
```

**危害：** 大量这类请求可以让 MySQL 过载，有时是恶意攻击（用不存在的 id 频繁请求）。

### 2.2 解决方案

**方案一：缓存空值**

当数据库查询结果为空时，也把空值写入 Redis，设置较短的过期时间。

```python
def get_user(user_id):
    key = f"user:{user_id}"
    cached = redis.get(key)
    
    if cached is not None:
        if cached == "NULL":
            return None   # 之前查过，确实不存在
        return json.loads(cached)
    
    user = db.query_user(user_id)
    if user is None:
        redis.setex(key, 300, "NULL")  # 缓存空值，5分钟过期
        return None
    
    redis.setex(key, 3600, json.dumps(user))
    return user
```

**优点：** 简单，实现成本低。

**缺点：**

- 如果攻击者用不同的不存在 id 攻击，会缓存大量空值，浪费 Redis 内存。
- 短暂不一致：如果数据后来被创建了，缓存里还是空值（直到过期）。

**方案二：布隆过滤器（Bloom Filter）**

在 Redis 前面加一层布隆过滤器，把所有存在的 id 预先加入过滤器，请求来了先判断 id 是否可能存在，不存在直接返回。

```text
请求(id=1001) → 布隆过滤器（可能存在）→ Redis → MySQL
请求(id=-1)   → 布隆过滤器（一定不存在）→ 直接返回，不查 Redis 和 MySQL
```

**布隆过滤器的特性：**

- **不存在时一定准确**：如果布隆过滤器说不存在，就一定不存在。
- **存在时可能误报**：说存在的有小概率是误报（假阳性），但概率可控（通常 < 1%）。
- 内存极小：10 亿个元素只需约 1.2GB（比 Set 小得多）。

```python
# 使用 Redis 的 Bitmap 实现简单布隆过滤器
# 或使用 Redisson、Bloom Filter 专用库

from redis import Redis
import mmh3  # MurmurHash

class BloomFilter:
    def __init__(self, redis, key, capacity=10_000_000, error_rate=0.01):
        self.redis = redis
        self.key = key
        self.size = self._calc_size(capacity, error_rate)
        self.hashes = self._calc_hashes(error_rate)
    
    def add(self, item):
        for seed in range(self.hashes):
            index = mmh3.hash(item, seed) % self.size
            self.redis.setbit(self.key, index, 1)
    
    def exists(self, item):
        for seed in range(self.hashes):
            index = mmh3.hash(item, seed) % self.size
            if not self.redis.getbit(self.key, index):
                return False  # 一定不存在
        return True  # 可能存在（有小概率误报）
```

**优点：** 内存效率高，即使 id 空间巨大也能高效过滤。

**缺点：** 有一定实现复杂度；数据增加时需要更新布隆过滤器；删除数据时无法从布隆过滤器中移除（标准布隆过滤器不支持删除）。

## 3. 缓存击穿（Cache Breakdown）

### 3.1 什么是缓存击穿

**定义：** 一个**热点 key** 过期的瞬间，大量并发请求同时涌入，全部打到数据库，造成数据库瞬间压力暴增。

```text
正常状态：
10000 个并发请求 → Redis（热点 key 命中）→ 全部从 Redis 返回

热点 key 过期的瞬间：
10000 个并发请求 → Redis（不命中，key 刚过期）
                → 10000 个请求全部打到 MySQL
                → MySQL 过载
```

**和缓存穿透的区别：**

- 缓存穿透：数据根本不存在，每次都穿透。
- 缓存击穿：数据存在，但热点 key 的那一刻过期，瞬间穿透。

### 3.2 解决方案

**方案一：互斥锁（Mutex Lock）**

同一时间只允许一个请求去查数据库，其他请求等待或者稍后重试。

```python
def get_hot_data(key):
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    
    # 尝试获取互斥锁
    lock_key = f"lock:{key}"
    if redis.set(lock_key, "1", nx=True, ex=10):
        try:
            # 获取锁成功，查数据库
            data = db.query(key)
            redis.setex(key, 3600, json.dumps(data))
            return data
        finally:
            redis.delete(lock_key)
    else:
        # 获取锁失败，稍等后重试（或者返回旧数据）
        time.sleep(0.05)
        return get_hot_data(key)
```

**优点：** 保证只有一个请求打到数据库，简单有效。

**缺点：** 锁等待期间响应延迟增加，高并发下大量请求排队。

**方案二：永不过期（逻辑过期）**

不给热点 key 设置 TTL，而是在 value 里记录"逻辑过期时间"，后台异步刷新。

```python
def get_hot_data(key):
    data = redis.get(key)
    if not data:
        # 第一次加载
        value = {"data": db.query(key), "expire_time": time.time() + 3600}
        redis.set(key, json.dumps(value))
        return value["data"]
    
    value = json.loads(data)
    
    if time.time() > value["expire_time"]:
        # 逻辑上已过期，但仍返回旧数据
        # 异步刷新
        if redis.set(f"lock:{key}", "1", nx=True, ex=10):
            threading.Thread(target=refresh_data, args=(key,)).start()
    
    return value["data"]  # 直接返回，不等待刷新

def refresh_data(key):
    new_data = db.query(key)
    value = {"data": new_data, "expire_time": time.time() + 3600}
    redis.set(key, json.dumps(value))
    redis.delete(f"lock:{key}")
```

**优点：** 用户请求永远不等待，响应最快，热点数据永不"击穿"。

**缺点：** 过期期间会返回旧数据，有短暂不一致；内存占用（热点 key 永不过期）。

## 4. 缓存雪崩（Cache Avalanche）

### 4.1 什么是缓存雪崩

**定义：** 大量缓存 key **在同一时间段集中过期**，或者 **Redis 服务宕机**，导致大量请求同时打到数据库，数据库扛不住崩溃。

```text
场景一：批量缓存同时过期
凌晨 0 点，你做了一次促销缓存预热，设置了 10000 个 key，TTL 都是 2 小时
凌晨 2 点：10000 个 key 同时过期
           所有请求同时打到 MySQL → MySQL 崩溃

场景二：Redis 服务宕机
Redis 宕机 → 所有请求直接打 MySQL → MySQL 崩溃
```

**和缓存击穿的区别：**

- 缓存击穿：一个热点 key 过期。
- 缓存雪崩：大量 key 同时过期，或 Redis 整体不可用。

### 4.2 解决方案

**方案一：过期时间加随机抖动**

设置过期时间时加上随机偏移量，让不同 key 的过期时间分散开来。

```python
import random

def cache_set(key, value, base_ttl=3600):
    # 在基础 TTL 上加 ±10% 的随机抖动
    ttl = base_ttl + random.randint(-base_ttl // 10, base_ttl // 10)
    redis.setex(key, ttl, json.dumps(value))
```

**方案二：Redis 高可用部署**

使用哨兵或 Cluster，Redis 宕机时自动切换，避免整体不可用导致的雪崩。

**方案三：多级缓存**

```text
本地缓存（L1，Guava/Caffeine）
    ↓ 不命中
Redis（L2）
    ↓ 不命中
数据库
```

即使 Redis 宕机，本地缓存还能扛一部分请求。

**方案四：熔断降级**

当数据库请求量超过阈值时，触发熔断，返回默认值或错误，避免数据库被压垮。

**方案五：限流**

在 Redis 缓存层之后、数据库之前加限流，控制打到数据库的请求数量。

## 5. 三大问题对比

| 问题 | 触发原因 | 影响范围 | 核心解决思路 |
|------|---------|---------|------------|
| 缓存穿透 | 查不存在的数据 | 持续性，每次都穿透 | 布隆过滤器 / 缓存空值 |
| 缓存击穿 | 热点 key 突然过期 | 瞬间，一个 key | 互斥锁 / 永不过期 |
| 缓存雪崩 | 大量 key 同时过期或 Redis 宕机 | 持续性，大面积 | 随机 TTL / 高可用 / 多级缓存 |

## 6. 缓存一致性

### 6.1 为什么会不一致

Redis 缓存是数据库数据的副本，当数据库数据更新时，如果缓存没有同步更新，就会出现不一致。

### 6.2 Cache-Aside 模式（旁路缓存，最常用）

**读操作：**

```text
先读 Redis → 命中则返回
不命中 → 读 MySQL → 写入 Redis → 返回
```

**写操作（推荐方式：更新数据库 + 删除缓存）：**

```text
更新 MySQL → 删除 Redis 中的缓存 key
下次读请求：读 Redis（不命中）→ 读 MySQL → 写入 Redis
```

**为什么是"删缓存"而不是"更新缓存"？**

删除比更新更安全：

```text
问题场景（先更新缓存，后更新数据库）：
线程A：更新缓存为新值 v2
线程B：更新缓存为旧值 v1（并发下可能逆序）
最终：缓存是 v1，数据库可能是 v2 → 不一致

删缓存不会有这个问题：
无论顺序如何，最终都会删掉缓存
下次读时重新从数据库加载，保证最终一致
```

### 6.3 先删缓存还是先更新数据库？

**推荐：先更新数据库，再删缓存。**

```text
先更新数据库，再删缓存（推荐）：

线程A：update MySQL（v→v2）→ delete Redis
问题：如果 delete Redis 失败（网络故障），缓存里还是旧值
解决：重试 / 订阅 binlog 异步删除

少见问题场景（理论上存在）：
线程A：update MySQL
同时线程B：read Redis（不命中）→ read MySQL（读到旧值，因为A还未提交？）→ write Redis（旧值）
线程A：delete Redis（删了，但B又写进去了旧值）
最终：缓存是旧值
→ 这种场景需要数据库延迟提交 + 消息队列异步删缓存才能完全解决
```

**先删缓存，再更新数据库（不推荐）：**

```text
线程A：delete Redis → (稍等) → update MySQL
线程B：read Redis（不命中）→ read MySQL（读到旧值，A还没写完）→ write Redis（旧值）
线程A：update MySQL 完成
最终：缓存是旧值，数据库是新值 → 不一致
```

### 6.4 延迟双删策略

为了减少不一致窗口，可以用延迟双删：

```python
def update_data(key, new_value):
    redis.delete(f"cache:{key}")     # 先删缓存
    db.update(key, new_value)        # 再更新数据库
    time.sleep(0.1)                  # 等待可能的并发读操作完成
    redis.delete(f"cache:{key}")     # 再删一次缓存（延迟双删）
```

延迟双删可以降低不一致的概率，但不能完全消除。

### 6.5 基于 Canal + MQ 的最终一致方案

更可靠的方案是订阅 MySQL binlog，异步删除缓存：

```text
MySQL 数据更新
   ↓ 产生 binlog
Canal 监听 binlog（阿里开源的 MySQL binlog 订阅组件）
   ↓
发送消息到 MQ
   ↓
消费者读取消息
   ↓
删除 Redis 缓存
```

这种方案与业务代码解耦，删缓存有重试保障，是生产环境常用的可靠方案。

## 7. 缓存预热

系统启动或大促活动前，提前把热点数据加载到 Redis，避免冷启动时大量请求打到 MySQL。

```python
def warm_up_cache():
    """系统启动时预热热点数据"""
    hot_products = db.query("SELECT * FROM product WHERE is_hot = 1")
    for product in hot_products:
        redis.setex(
            f"product:{product.id}",
            3600,
            json.dumps(product.to_dict())
        )
```

## 8. 面试回答模板

缓存三大问题：

**缓存穿透**：查询不存在的数据，缓存和数据库都没有，每次都打到数据库。解决方案：布隆过滤器提前过滤不存在的 key，或者缓存空值（设置短过期时间）。

**缓存击穿**：热点 key 过期瞬间，大量并发请求同时打到数据库。解决方案：互斥锁（同时只有一个请求查数据库）或逻辑过期（不设置 TTL，异步刷新，始终返回缓存）。

**缓存雪崩**：大量 key 同时过期或 Redis 宕机，大量请求同时打到数据库。解决方案：给 TTL 加随机抖动、部署高可用 Redis、多级缓存、限流降级。

缓存一致性：推荐 Cache-Aside 模式，先更新数据库再删除缓存。更可靠的方案是通过 Canal 订阅 binlog，异步删除缓存。

## 我应该背下来的最短答案

穿透（查不存在的数据）→ 布隆过滤器或缓存空值；击穿（热点key过期）→ 互斥锁或逻辑过期；雪崩（大量key同时过期）→ 随机TTL抖动 + 高可用部署。缓存一致性推荐先更新DB再删缓存，可配合binlog异步删除保证可靠性。
