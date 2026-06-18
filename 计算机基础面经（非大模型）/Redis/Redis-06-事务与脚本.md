---
type: knowledge
status: active
created: 2026-06-12
updated: 2026-06-12
tags:
  - 面试/数据库
  - Redis
aliases:
  - Redis事务
  - Redis Lua脚本
  - MULTI EXEC
---

# Redis 事务与脚本

## 1. 面试官到底想考什么

- 你是否知道 Redis 事务和 MySQL 事务的核心区别（Redis 不支持回滚）。
- 你能否说清楚 MULTI/EXEC 的流程和局限。
- 你是否了解 Lua 脚本是如何保证原子性的，以及它和事务的区别。

## 2. Redis 事务基础

### 2.1 Redis 事务的命令

```bash
MULTI     # 开始事务，进入排队模式
命令1      # 命令不立即执行，被放入队列
命令2
命令3
EXEC      # 提交事务，队列中的命令依次执行
DISCARD   # 取消事务，清空命令队列
```

### 2.2 事务执行流程

```text
客户端: MULTI
服务端: OK
客户端: SET k1 v1
服务端: QUEUED     ← 排队，不执行
客户端: INCR counter
服务端: QUEUED
客户端: SET k2 v2
服务端: QUEUED
客户端: EXEC
服务端: 1) OK      ← k1 设置成功
        2) (integer) 1  ← counter 自增成功
        3) OK      ← k2 设置成功
```

### 2.3 Redis 事务 vs MySQL 事务

这是面试必问的对比：

| 特性 | Redis 事务 | MySQL 事务 |
|------|-----------|-----------|
| 原子性 | 弱（不支持回滚） | 强（支持回滚） |
| 隔离性 | 有（执行期间不会被其他命令插入） | 有（多种隔离级别） |
| 持久性 | 取决于持久化配置 | 支持（WAL/redo log） |
| 一致性 | 部分（执行时错误不回滚） | 支持 |
| 回滚 | 不支持 | 支持 |

**核心区别：Redis 事务不支持回滚。**

```text
MULTI
SET k1 v1
INCR string_key    ← 这条命令运行时会出错（string_key 不是整数）
SET k2 v2
EXEC
```

执行结果：

```text
1) OK          ← SET k1 v1 成功
2) ERR         ← INCR 出错（运行时错误）
3) OK          ← SET k2 v2 成功！
```

**注意：** 运行时错误不会导致其他命令回滚，前后命令照常执行。

### 2.4 两种错误的处理方式不同

**语法错误（编译时错误）** —— 整个事务不执行：

```text
MULTI
SET k1 v1
WRONGCMD   ← 语法错误（命令不存在）
SET k2 v2
EXEC
```

结果：EXEC 返回错误，整个事务被取消，k1 和 k2 都不会被设置。

**运行时错误** —— 错误命令失败，其他命令照常执行：

```text
MULTI
SET k1 v1
INCR string_key   ← 运行时错误
SET k2 v2
EXEC
```

结果：k1 和 k2 都被设置，只有 INCR 那条命令报错。

## 3. WATCH：乐观锁

### 3.1 WATCH 的作用

WATCH 提供 **CAS（Compare-And-Swap）** 语义，用于解决事务期间数据被其他客户端修改的问题。

```bash
WATCH key [key ...]   # 监视一个或多个键
```

如果在 MULTI/EXEC 期间，被 WATCH 的键被其他客户端修改了，EXEC 会返回 nil（事务执行失败）。

### 3.2 WATCH 使用示例

**场景：** 购票系统，检查余票后购买，防止超卖。

```python
def buy_ticket(ticket_id, user_id):
    while True:
        # WATCH 余票数量
        redis.watch(f"ticket:{ticket_id}:count")
        count = int(redis.get(f"ticket:{ticket_id}:count") or 0)
        if count <= 0:
            redis.unwatch()
            return False  # 无票
        
        # 开启事务
        pipe = redis.pipeline(True)
        pipe.multi()
        pipe.decr(f"ticket:{ticket_id}:count")       # 余票 -1
        pipe.sadd(f"ticket:{ticket_id}:buyers", user_id)  # 记录购买者
        try:
            pipe.execute()  # 如果 count 被其他人改过，这里会抛异常
            return True  # 购票成功
        except redis.WatchError:
            continue  # 重试
```

### 3.3 WATCH 的局限

- 需要客户端重试逻辑，高并发下可能重试次数很多，性能下降。
- 适合低并发场景，高并发场景建议用 Lua 脚本替代。

## 4. Lua 脚本

### 4.1 为什么用 Lua 脚本

Lua 脚本在 Redis 中是**原子执行**的：执行期间，Redis 不会处理其他命令，保证了脚本的原子性。

相比事务，Lua 脚本的优势：

- 可以在脚本内做判断（if/else 逻辑）。
- 原子性更强（整个脚本作为一个原子操作）。
- 支持复杂业务逻辑，不需要客户端往返多次。

### 4.2 基本语法

```bash
EVAL script numkeys [key [key ...]] [arg [arg ...]]
```

- `script`：Lua 脚本内容。
- `numkeys`：key 的数量。
- 脚本内用 `KEYS[1]`, `KEYS[2]`... 访问键，用 `ARGV[1]`, `ARGV[2]`... 访问参数。

### 4.3 常用 Lua 脚本示例

**示例一：原子性的检查并设置**

```lua
-- 如果 key 的值等于 expected，才更新为 new_value
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2])
    return 1
else
    return 0
end
```

```bash
EVAL "local current = redis.call('GET', KEYS[1]); if current == ARGV[1] then redis.call('SET', KEYS[1], ARGV[2]); return 1 else return 0 end" 1 mykey old_value new_value
```

**示例二：分布式锁释放（原子判断 + 删除）**

```lua
-- 只有 value 匹配才删除，防止删掉别人的锁
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
```

```python
release_lock_script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""
redis.eval(release_lock_script, 1, lock_key, lock_value)
```

**示例三：原子限流**

```lua
-- 滑动窗口限流
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end
if current > limit then
    return 0  -- 超限
else
    return 1  -- 允许
end
```

### 4.4 脚本缓存

每次发送完整脚本内容比较浪费网络带宽。Redis 支持脚本缓存：

```bash
SCRIPT LOAD "return redis.call('GET', KEYS[1])"
# 返回 SHA1 校验值，如 "e0e1f9fabfa9d353eca4c6f5d9606c3f7abb56b2"

EVALSHA e0e1f9fabfa9d353eca4c6f5d9606c3f7abb56b2 1 mykey
# 通过 SHA1 调用脚本，不需要每次发送完整脚本
```

```bash
SCRIPT EXISTS sha1 [sha1 ...]   # 检查脚本是否已缓存
SCRIPT FLUSH                    # 清空所有缓存的脚本
```

### 4.5 Lua 脚本的注意事项

- 脚本执行期间 Redis 不处理其他命令，脚本不能太慢（避免阻塞）。
- 脚本中的随机命令（如 RANDOMKEY、SRANDMEMBER）结果是不确定的，AOF 重放时可能不一致，需要注意。
- Redis 7.0 之前，脚本错误中途退出不会回滚已执行的部分。Redis 7.0+ 通过 MULTI 封装脚本可以支持回滚。

## 5. Pipeline（管道）

虽然 Pipeline 不是严格意义上的"事务"，但经常和事务一起讨论，因为它也是批量发送命令的方式。

### 5.1 Pipeline vs 事务

| 对比 | Pipeline | 事务（MULTI/EXEC） |
|------|---------|-----------------|
| 原子性 | 无（命令可能被其他客户端插入） | 有（EXEC 时命令顺序执行） |
| 减少网络往返 | 是（批量发送） | 是（EXEC 后一次性执行） |
| 目的 | 减少网络延迟，提升吞吐 | 保证命令的原子性 |
| 服务端处理 | 逐条处理，非原子 | EXEC 时批量处理，原子 |

**Pipeline 的典型用法：**

```python
# 批量写入，减少网络往返次数
pipe = redis.pipeline(transaction=False)  # transaction=False 不开启 MULTI/EXEC
for i in range(1000):
    pipe.set(f"key:{i}", f"value:{i}")
pipe.execute()  # 一次性发送 1000 条命令，只有 1 次网络往返
```

## 6. 面试回答模板

Redis 事务通过 MULTI/EXEC 实现，把多条命令放入队列后一次性执行，期间不会被其他命令插入。但 Redis 事务不支持回滚：语法错误会取消整个事务，但运行时错误只影响那条命令，前后命令照常执行。这是和 MySQL 事务最大的区别。

WATCH 提供乐观锁，如果被监视的键在 EXEC 前被其他客户端修改，整个事务取消，适合低并发的 CAS 场景。

Lua 脚本是更强的原子性方案，脚本执行期间 Redis 不处理其他命令，并且脚本内可以有条件判断，比事务更灵活，常用于分布式锁释放、复杂原子操作等场景。

## 我应该背下来的最短答案

Redis 事务（MULTI/EXEC）不支持回滚，运行时错误只影响那条命令，其他命令正常执行；WATCH 提供乐观锁；Lua 脚本原子执行，且支持条件逻辑，是更推荐的原子操作方式。
