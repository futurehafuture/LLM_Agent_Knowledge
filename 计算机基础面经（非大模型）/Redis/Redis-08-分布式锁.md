---
type: knowledge
status: active
created: 2026-06-12
updated: 2026-06-12
tags:
  - 面试/数据库
  - Redis
aliases:
  - Redis分布式锁
  - Redlock
  - SETNX
---

# Redis 分布式锁

## 1. 面试官到底想考什么

分布式锁是面试高频题，面试官想知道：

- 你是否理解为什么需要分布式锁（Java synchronized 不够用）。
- 你能否说出用 Redis 实现分布式锁的演进过程，以及每个版本解决了什么问题。
- 你是否了解 Redlock 算法以及争议。

## 2. 为什么需要分布式锁

### 2.1 单机锁的局限

在单机程序里，用 synchronized 或 ReentrantLock 可以防止多线程并发问题：

```java
// 单机，一个 JVM 内，这样没问题
synchronized (this) {
    // 检查库存，扣减库存
}
```

但在**分布式系统**中，同一个服务部署了多台机器，每台机器有自己的 JVM，synchronized 只能锁住本机的线程，不同机器之间无法互斥：

```text
机器A：JVM锁 → 扣减库存
机器B：同时：JVM锁 → 扣减库存  ← 两台机器同时操作，超卖！
```

**分布式锁**就是一个"跨机器"的互斥机制，让多台服务器竞争同一把锁。

### 2.2 需要分布式锁的典型场景

- **防止超卖**：多台服务器同时处理同一商品的库存扣减。
- **定时任务幂等**：多台机器的定时任务，只允许一台机器执行。
- **接口防重复提交**：同一用户的同一请求只处理一次。
- **资源竞争**：多个服务节点抢占同一个处理任务。

## 3. 分布式锁的基本要求

一个可靠的分布式锁必须满足：

1. **互斥性**：同一时刻只有一个客户端持有锁。
2. **避免死锁**：即使持有锁的客户端崩溃，锁也能被释放（有过期时间）。
3. **只能自己释放自己的锁**：不能把别人的锁删掉。
4. **高可用**：锁服务本身不能成为单点故障。

## 4. 基于 Redis 的分布式锁演进

### 4.1 第一版：SETNX + EXPIRE（有问题）

```bash
SETNX lock:resource_id 1     # 设置锁（NX = Not eXists，key不存在时才设置）
EXPIRE lock:resource_id 30   # 设置过期时间（防止死锁）
```

**致命问题：** SETNX 和 EXPIRE 是两条命令，不是原子操作！

```text
线程A：SETNX 成功 → 准备 EXPIRE
此时：线程A 崩溃（或服务器宕机）
EXPIRE 没有执行
lock 永远不会过期 → 死锁！
```

### 4.2 第二版：SET NX PX（正确方式）

Redis 2.6.12+ 支持在 SET 命令里同时指定 NX 和 PX（过期时间），这是**原子操作**。

```bash
SET lock:resource_id unique_value NX PX 30000
# NX：key不存在时才设置
# PX 30000：30000 毫秒后过期
# unique_value：唯一标识（谁加的锁）
```

**加锁：**

```python
import uuid

def acquire_lock(redis, resource, expire_ms=30000):
    lock_key = f"lock:{resource}"
    lock_value = str(uuid.uuid4())    # 唯一标识，防止误删他人的锁
    
    result = redis.set(lock_key, lock_value, nx=True, px=expire_ms)
    if result:
        return lock_value  # 返回锁的唯一标识，释放时需要用
    return None  # 加锁失败（锁已被占用）
```

**释放锁（必须用 Lua 脚本保证原子性）：**

```python
RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""

def release_lock(redis, resource, lock_value):
    lock_key = f"lock:{resource}"
    result = redis.eval(RELEASE_LOCK_SCRIPT, 1, lock_key, lock_value)
    return result == 1
```

**为什么释放锁要用 Lua 脚本？**

GET 和 DEL 必须是原子操作，否则会有问题：

```text
线程A：GET lock（确认是自己的值）
       ← 锁过期，线程B 加锁成功 ←
线程A：DEL lock（把线程B的锁删了！）
```

Lua 脚本在 Redis 中是原子执行的，GET 和 DEL 不会被打断。

### 4.3 锁续期问题（Watch Dog 机制）

如果业务执行时间超过锁的过期时间，锁会自动释放，其他客户端可以加锁，导致多个客户端同时持有"锁"。

**Redisson 的 Watch Dog 机制：**

Redisson 是 Redis 的 Java 客户端，它实现了自动续期：

```text
加锁时，默认 TTL = 30 秒
后台线程每 10 秒检查一次：如果锁还被当前客户端持有，就续期到 30 秒
业务完成，主动释放锁，Watch Dog 停止
```

```java
// Redisson 使用示例
RLock lock = redisson.getLock("lock:resource");
try {
    lock.lock();  // 自动续期，不需要指定超时时间
    // 或
    lock.lock(10, TimeUnit.SECONDS);  // 指定超时时间，不会自动续期
    // 业务代码
} finally {
    lock.unlock();
}
```

### 4.4 完整的生产可用实现

```python
import uuid
import time
import threading

class RedisDistributedLock:
    
    ACQUIRE_SCRIPT = """return redis.call('SET', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2])"""
    
    RELEASE_SCRIPT = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('DEL', KEYS[1])
    else
        return 0
    end
    """
    
    RENEW_SCRIPT = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('PEXPIRE', KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    
    def __init__(self, redis, resource, expire_ms=30000, retry_times=3, retry_delay=0.1):
        self.redis = redis
        self.lock_key = f"lock:{resource}"
        self.lock_value = str(uuid.uuid4())
        self.expire_ms = expire_ms
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self._watchdog = None
    
    def acquire(self):
        for _ in range(self.retry_times):
            result = self.redis.eval(
                self.ACQUIRE_SCRIPT, 1,
                self.lock_key, self.lock_value, self.expire_ms
            )
            if result:
                self._start_watchdog()
                return True
            time.sleep(self.retry_delay)
        return False
    
    def release(self):
        self._stop_watchdog()
        return self.redis.eval(
            self.RELEASE_SCRIPT, 1,
            self.lock_key, self.lock_value
        ) == 1
    
    def _start_watchdog(self):
        """每隔 expire/3 时间续期一次"""
        interval = self.expire_ms / 3000  # 毫秒转秒，取1/3
        self._watchdog = threading.Timer(interval, self._renew_and_restart)
        self._watchdog.daemon = True
        self._watchdog.start()
    
    def _renew_and_restart(self):
        result = self.redis.eval(
            self.RENEW_SCRIPT, 1,
            self.lock_key, self.lock_value, self.expire_ms
        )
        if result:  # 续期成功，继续监控
            self._start_watchdog()
    
    def _stop_watchdog(self):
        if self._watchdog:
            self._watchdog.cancel()
    
    def __enter__(self):
        if not self.acquire():
            raise Exception("Failed to acquire lock")
        return self
    
    def __exit__(self, *args):
        self.release()
```

## 5. Redlock 算法

### 5.1 单机 Redis 分布式锁的问题

如果 Redis 是单节点，Redis 挂了就无法加锁；如果是主从架构，会有数据同步延迟问题：

```text
线程A 在主节点加锁成功
主节点把锁同步给从节点之前，主节点宕机
哨兵把从节点提升为新主节点
新主节点上没有这把锁
线程B 在新主节点加锁成功
→ 线程A 和线程B 同时持有锁！
```

### 5.2 Redlock 算法

为了解决这个问题，Redis 作者 Antirez 提出了 **Redlock 算法**，基于多个独立的 Redis 节点（通常 5 个）：

```text
准备 5 个独立的 Redis 节点（不是主从，不是集群）

加锁步骤：
1. 记录加锁开始时间 T1
2. 依次向 5 个节点发送 SET NX PX 命令（每个节点设置较短的超时时间，如 150ms）
3. 统计成功加锁的节点数
4. 计算加锁总耗时 T = now - T1

加锁成功条件：
- 成功节点数 >= N/2 + 1（5个节点需要3个成功）
- 加锁总耗时 T < 锁的过期时间

锁的有效时间 = 原始过期时间 - 加锁耗时 T（要用剩余的有效时间）

释放锁：向所有节点发送释放命令（即使加锁失败的节点也要释放，防止网络问题导致的意外加锁）
```

### 5.3 Redlock 的争议

Redlock 算法存在一定争议，著名的有 Martin Kleppmann（《设计数据密集型应用》作者）和 Antirez 之间的论战：

**Martin 的质疑：**

- 分布式系统中，时钟可能不准（NTP 跳跃），过期时间不可靠。
- 进程可能在释放锁之前发生"停顿"（GC、网络延迟），导致锁已过期但客户端还以为自己持有锁。
- 如果需要绝对安全，应该使用 Fencing Token（单调递增的 token）+ 数据库乐观锁。

**Antirez 的反驳：**

- Redlock 的设计目标是"效率锁"（减少重复工作），而非"正确性锁"（保证互斥）。
- 在合理的时钟偏差假设下，Redlock 是足够可靠的。

**实际建议：**

- 如果只是防止重复处理（接口幂等、定时任务），单节点 Redis 锁 + 业务层幂等校验就够了。
- 如果对正确性要求极高，考虑 ZooKeeper、etcd 这类专门的分布式协调服务。
- Redlock 适合中间场景：比单节点更可靠，但也不需要 ZooKeeper 的复杂性。

## 6. 面试回答模板

分布式锁是为了解决多台服务器并发操作同一资源的互斥问题，单机的 synchronized 在分布式环境下无效。

用 Redis 实现分布式锁，核心是用 `SET key value NX PX expire` 命令（原子加锁 + 过期时间），value 用 UUID 唯一标识锁的持有者。释放锁时，必须用 Lua 脚本原子地"判断 value 是自己的，才删除"，防止误删他人的锁。

为了防止业务时间超过锁过期时间，可以实现 Watch Dog 机制（Redisson 实现了这个），每隔一段时间自动续期。

Redlock 算法通过多个独立 Redis 节点投票加锁，解决单节点/主从架构下的可靠性问题，但存在时钟依赖等争议，适合中等可靠性要求的场景。

## 我应该背下来的最短答案

Redis 分布式锁：用 `SET lock_key uuid NX PX 30000`（原子加锁+过期），释放时用 Lua 脚本判断 value 是否匹配再删除（防误删）。防止超时可用 Watch Dog 自动续期（Redisson 实现）。Redlock 算法基于多个独立 Redis 节点投票，解决单点问题，但有时钟依赖争议。
