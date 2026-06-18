---
type: knowledge
status: active
created: 2026-06-12
updated: 2026-06-12
tags:
  - 面试/数据库
  - Redis
aliases:
  - Redis数据类型
  - Redis五种数据类型
---

# Redis 数据类型

## 1. 面试官到底想考什么

面试官问数据类型，通常不只想听"有五种"，而是想知道：

- 你能否说出每种类型的**典型使用场景**。
- 你是否知道它们的**底层实现**（进阶追问）。
- 你是否能根据场景选择最合适的数据类型，而不是什么都用 String。

## 2. 五种核心数据类型总览

| 类型 | 中文 | 典型命令 | 典型场景 |
|------|------|---------|---------|
| String | 字符串 | GET/SET/INCR | 缓存、计数器、分布式锁 |
| List | 列表 | LPUSH/RPOP/LRANGE | 消息队列、最新动态列表 |
| Hash | 哈希表 | HSET/HGET/HGETALL | 对象存储、购物车 |
| Set | 集合 | SADD/SMEMBERS/SINTER | 去重、标签、共同好友 |
| Sorted Set | 有序集合 | ZADD/ZRANGE/ZREVRANK | 排行榜、延迟队列 |

Redis 还有三种特殊类型：**Bitmap、HyperLogLog、Geo**，以及 Redis 5.0 引入的 **Stream**。

## 3. String（字符串）

### 3.1 基本介绍

String 是 Redis 最基础的类型，一个 key 对应一个值。值可以是：

- 普通字符串（如 `"hello"`）
- 整数（如 `42`）——支持原子自增/自减
- 二进制数据（如序列化的 JSON 或图片字节）

最大容量：**512 MB**。

### 3.2 常用命令

```bash
# 基本操作
SET key value                      # 设置值
GET key                            # 获取值
DEL key                            # 删除键
EXISTS key                         # 判断是否存在

# 带过期时间
SET key value EX 3600              # 设置值，3600 秒后过期
SET key value PX 1000              # 毫秒级过期
SETEX key 3600 value               # 同上，旧式写法
GETEX key EX 3600                  # 获取值并刷新过期时间

# 仅当 key 不存在时才设置（常用于分布式锁）
SET key value NX                   # NX = Not eXists
SET key value NX EX 30             # 不存在时设置，30秒过期

# 数值操作（要求 value 是整数）
INCR key                           # 值 +1（原子操作）
INCRBY key 5                       # 值 +5
DECR key                           # 值 -1
DECRBY key 5                       # 值 -5
INCRBYFLOAT key 1.5                # 浮点数加法

# 批量操作
MSET k1 v1 k2 v2 k3 v3            # 批量设置
MGET k1 k2 k3                     # 批量获取
```

### 3.3 使用场景

**场景一：缓存**

```python
# 把 MySQL 查询结果序列化后存入 Redis
import json

def get_user_info(user_id):
    key = f"user:info:{user_id}"
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    user = db.query_user(user_id)
    redis.setex(key, 3600, json.dumps(user))
    return user
```

**场景二：计数器**

```bash
# 文章点击量，每次请求自增（原子操作，不会出现并发问题）
INCR article:click:10086
```

**场景三：分布式锁**

```bash
# SET NX EX 是原子操作：只有 key 不存在时才设置，设置时带过期时间
SET lock:order:5001 "thread-uuid-xxx" NX EX 30
```

**场景四：限流**

```python
# 每分钟最多 100 次请求
def is_rate_limited(user_id):
    key = f"rate:{user_id}:{int(time.time() // 60)}"
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, 60)
    return count > 100
```

### 3.4 底层实现

Redis 的 String 底层是 **SDS（Simple Dynamic String，简单动态字符串）**，而非 C 的原生字符串。

SDS 优势：
- 记录字符串长度（O(1) 获取长度，C 字符串需 O(n)）。
- 预留空间，追加时不需要频繁重分配内存。
- 二进制安全，可以存储任意字节（包括 `\0`）。

对于整数值，Redis 会进一步用 **int 编码**直接存储，更节省内存。

## 4. List（列表）

### 4.1 基本介绍

List 是有序的字符串列表，**元素可以重复**，支持从两端插入和删除。类似 Java 的 LinkedList（双向链表）。

特点：
- 插入删除快（O(1)），随机访问慢（O(n)）。
- 支持阻塞弹出，可以做消息队列。
- 元素有序（按插入顺序）。

### 4.2 常用命令

```bash
# 插入
LPUSH key v1 v2 v3              # 从左端插入（头部），最后 v3 在最左边
RPUSH key v1 v2 v3              # 从右端插入（尾部）
LINSERT key BEFORE/AFTER pivot value  # 在指定元素前/后插入

# 弹出
LPOP key                        # 从左端弹出（移除并返回）
RPOP key                        # 从右端弹出
BLPOP key timeout               # 阻塞左端弹出，timeout=0 表示永久等待
BRPOP key timeout               # 阻塞右端弹出

# 查询
LRANGE key start stop           # 获取指定范围（0 -1 获取全部）
LINDEX key index                # 获取指定下标
LLEN key                        # 获取列表长度

# 修改
LSET key index value            # 修改指定下标的值
LREM key count value            # 删除指定值（count>0 从头删，count<0 从尾删）
LTRIM key start stop            # 裁剪列表（只保留指定范围）
```

### 4.3 使用场景

**场景一：消息队列（简单版）**

```bash
# 生产者
RPUSH task_queue "task:123"

# 消费者（阻塞等待，有消息立即返回）
BRPOP task_queue 0
```

**场景二：最新动态列表（时间线）**

```bash
# 微博/朋友圈，每发一条推入列表头部
LPUSH user:1001:timeline "post:9988"
# 只保留最新 100 条
LTRIM user:1001:timeline 0 99
# 取最新 20 条
LRANGE user:1001:timeline 0 19
```

**场景三：栈和队列**

```text
栈（LIFO）：LPUSH + LPOP
队列（FIFO）：RPUSH + LPOP
```

### 4.4 底层实现

List 的底层根据元素数量动态切换：

- **元素数量少（≤128），且每个元素长度短（≤64字节）**：使用 **listpack**（Redis 7.0+，之前叫 ziplist，压缩列表），内存连续，紧凑存储。
- **超出阈值**：转为 **quicklist**（快速链表，由多个 listpack 节点组成的双向链表）。

## 5. Hash（哈希表）

### 5.1 基本介绍

Hash 是一个键对应多个字段-值对的结构，类似 Java 的 HashMap，或者说是"字典的字典"。

适合存储**对象**（如用户信息），字段可以单独更新，不需要把整个对象序列化。

### 5.2 常用命令

```bash
# 设置
HSET key field value               # 设置单个字段
HSET key f1 v1 f2 v2 f3 v3        # 设置多个字段（Redis 4.0+）
HSETNX key field value             # 字段不存在时才设置

# 获取
HGET key field                     # 获取单个字段
HGETALL key                        # 获取所有字段和值
HMGET key f1 f2 f3                 # 批量获取字段
HKEYS key                          # 获取所有字段名
HVALS key                          # 获取所有值
HLEN key                           # 获取字段数量

# 数值操作
HINCRBY key field increment        # 整数字段自增
HINCRBYFLOAT key field increment   # 浮点字段自增

# 其他
HEXISTS key field                  # 判断字段是否存在
HDEL key field [field ...]         # 删除字段
```

### 5.3 使用场景

**场景一：存储对象（推荐！）**

```bash
# 存用户信息，字段可以单独更新
HSET user:1001 name "Alice" age 25 city "Shanghai" score 9800
HGET user:1001 name             # → Alice
HSET user:1001 score 9900       # 只更新 score，其他字段不变
```

对比用 String 存 JSON：

```text
String 存 JSON：更新 age 需要 GET → 修改 JSON → SET，多一次读操作
Hash：HSET user:1001 age 26 直接更新，不影响其他字段
```

**场景二：购物车**

```bash
# cart:{user_id} 这个 hash，field 是 product_id，value 是数量
HSET cart:1001 product:5001 2     # 加入商品 5001，数量 2
HINCRBY cart:1001 product:5001 1  # 数量 +1
HDEL cart:1001 product:5001       # 删除商品
HGETALL cart:1001                 # 获取购物车全部商品
HLEN cart:1001                    # 购物车商品种类数
```

**场景三：统计聚合**

```bash
# 网站各页面的访问统计
HINCRBY page_stats today /home 1
HINCRBY page_stats today /about 1
HGETALL page_stats today          # 获取今日所有页面统计
```

### 5.4 底层实现

- **元素少（≤128），且每个 field/value 长度短（≤64字节）**：使用 **listpack** 存储（紧凑，节省内存）。
- **超出阈值**：转为 **hashtable**（真正的哈希表，链式哈希）。

## 6. Set（集合）

### 6.1 基本介绍

Set 是**无序**、**不重复**的字符串集合。特点：

- 元素唯一，自动去重。
- 无序，不能按索引访问。
- 支持集合运算（交集、并集、差集）。

### 6.2 常用命令

```bash
# 基本操作
SADD key v1 v2 v3              # 添加元素（重复自动忽略）
SREM key v1 v2                 # 删除元素
SISMEMBER key value            # 判断元素是否在集合中
SMEMBERS key                   # 获取所有元素（无序）
SCARD key                      # 获取集合大小
SPOP key [count]               # 随机弹出元素（移除）
SRANDMEMBER key [count]        # 随机获取元素（不移除）

# 集合运算
SINTER key1 key2               # 交集
SUNION key1 key2               # 并集
SDIFF key1 key2                # 差集（key1 有但 key2 没有）
SINTERSTORE dest key1 key2     # 交集结果存入 dest
SUNIONSTORE dest key1 key2     # 并集结果存入 dest
SDIFFSTORE dest key1 key2      # 差集结果存入 dest
```

### 6.3 使用场景

**场景一：标签系统**

```bash
# 文章的标签
SADD article:1001:tags "Redis" "数据库" "缓存"
SADD article:1002:tags "MySQL" "数据库" "索引"

# 查询同时包含"数据库"标签的文章（需要逆向索引）
SADD tag:数据库:articles 1001 1002
SMEMBERS tag:数据库:articles
```

**场景二：共同好友 / 共同关注**

```bash
SADD user:alice:following bob charlie dave
SADD user:bob:following alice charlie evan

# alice 和 bob 的共同关注
SINTER user:alice:following user:bob:following  # → charlie
```

**场景三：抽奖 / 随机推荐**

```bash
# 奖池里加入用户
SADD lottery:pool "user:1001" "user:1002" "user:1003"
# 随机抽取 3 个中奖者（移除，确保不重复中奖）
SPOP lottery:pool 3
```

**场景四：去重访客统计**

```bash
# 今日 UV 统计
SADD uv:2026-06-12 "user:1001" "user:1002"
SADD uv:2026-06-12 "user:1001"   # 重复访问自动忽略
SCARD uv:2026-06-12              # 获取今日独立访客数
```

### 6.4 底层实现

- **元素全是整数且数量少（≤512）**：使用 **intset**（有序整数数组，二分查找）。
- **其他情况**：使用 **listpack**（元素少）或 **hashtable**（元素多）。

## 7. Sorted Set（有序集合）

### 7.1 基本介绍

Sorted Set（ZSet）是**有序**、**不重复**的字符串集合，每个元素关联一个 **score（分数）**，元素按 score 从小到大排序。

特点：

- 元素唯一（但 score 可以重复）。
- 按 score 自动排序。
- 支持按分数范围查询。
- 支持获取元素的排名。

### 7.2 常用命令

```bash
# 添加/更新
ZADD key score member                     # 添加元素
ZADD key NX score member                  # 只在不存在时添加
ZADD key XX score member                  # 只在存在时更新
ZADD key INCR 5 member                    # score 加 5

# 删除
ZREM key member [member ...]              # 删除元素
ZREMRANGEBYRANK key start stop            # 按排名范围删除
ZREMRANGEBYSCORE key min max              # 按分数范围删除

# 查询（升序）
ZRANGE key start stop [WITHSCORES]        # 按排名范围查询（低分 → 高分）
ZRANGEBYSCORE key min max [WITHSCORES]    # 按分数范围查询
ZRANGEBYLEX key min max                   # 按字典序范围查询（score 相同时）

# 查询（降序）
ZREVRANGE key start stop [WITHSCORES]     # 按排名反向查询（高分 → 低分）
ZREVRANGEBYSCORE key max min [WITHSCORES] # 按分数反向查询

# 排名
ZRANK key member                          # 获取排名（从0开始，低分在前）
ZREVRANK key member                       # 获取反向排名（高分在前）

# 其他
ZSCORE key member                         # 获取 score
ZCARD key                                 # 获取元素总数
ZCOUNT key min max                        # 统计分数范围内的元素数
ZINCRBY key increment member              # 增加 score

# 集合运算
ZUNIONSTORE dest numkeys key [key ...]    # 并集
ZINTERSTORE dest numkeys key [key ...]    # 交集
```

### 7.3 使用场景

**场景一：排行榜**

```bash
# 游戏积分排行榜
ZADD leaderboard 9800 "alice"
ZADD leaderboard 9500 "bob"
ZADD leaderboard 10200 "charlie"

# 获取前 10 名（降序，高分在前）
ZREVRANGE leaderboard 0 9 WITHSCORES
# → charlie 10200, alice 9800, bob 9500

# alice 的排名（从第0名开始）
ZREVRANK leaderboard "alice"
# → 1（第2名）

# 给 alice 加 200 分
ZINCRBY leaderboard 200 "alice"
```

**场景二：延迟队列**

```bash
# score 设为执行时间戳
ZADD delay_queue 1718380800 "task:123"   # 定时在 1718380800 执行
ZADD delay_queue 1718384400 "task:124"

# 消费者轮询：取出当前时间之前到期的任务
ZRANGEBYSCORE delay_queue 0 (当前时间戳) LIMIT 0 10
# 处理完后删除
ZREM delay_queue "task:123"
```

**场景三：热搜词**

```bash
# 每次搜索增加权重
ZINCRBY hot_search 1 "Redis面试题"
# 获取热搜 top 10
ZREVRANGE hot_search 0 9 WITHSCORES
```

**场景四：范围查询（价格区间）**

```bash
# 商品按价格索引
ZADD products:price 99.0 "product:1001"
ZADD products:price 299.0 "product:1002"
ZADD products:price 599.0 "product:1003"

# 查询 100 到 500 之间的商品
ZRANGEBYSCORE products:price 100 500
```

### 7.4 底层实现

Sorted Set 根据数据量使用两种底层结构：

- **元素数量少（≤128），且每个元素长度短（≤64字节）**：使用 **listpack**，节省内存。
- **超出阈值**：使用 **skiplist（跳表）+ hashtable** 双结构。
  - skiplist 维护有序结构，支持 O(log N) 的范围查询和插入删除。
  - hashtable 维护 member → score 的映射，支持 O(1) 的 ZSCORE 查询。

**跳表（Skip List）为什么不用红黑树？**

| 对比点 | 跳表 | 红黑树 |
|--------|------|--------|
| 实现复杂度 | 简单 | 复杂（旋转、变色） |
| 范围查询 | 天然支持，找到最小元素后顺序遍历 | 需要中序遍历 |
| 并发扩展 | 易于无锁实现 | 难 |
| 空间 | 稍多（索引层） | 稍少 |

Redis 选跳表是因为：**范围查询场景多，实现简单，性能和红黑树相近**。

## 8. 特殊数据类型

### 8.1 Bitmap（位图）

本质是 String，但把每一位（bit）作为操作单位。适合大量布尔值的存储。

```bash
# 用户签到（用户ID作为偏移量）
SETBIT signin:2026-06-12 1001 1    # 用户1001今天签到
GETBIT signin:2026-06-12 1001      # 查询签到状态 → 1
BITCOUNT signin:2026-06-12         # 今日签到总人数

# 1亿用户的在线状态，只需 1亿/8 ≈ 12.5MB
SETBIT online_users 1001 1
BITCOUNT online_users              # 在线人数
```

### 8.2 HyperLogLog

用于**近似统计**不重复元素的数量（基数统计）。误差率约 0.81%，但内存占用极小（固定 12KB）。

```bash
PFADD uv_counter "user:1001" "user:1002" "user:1001"   # 添加（重复自动忽略）
PFCOUNT uv_counter           # 近似去重统计 → 2
PFMERGE result uv1 uv2       # 合并多个 HyperLogLog
```

**适用场景**：网页 UV（独立访客数）统计，不需要精确结果时。比 Set 节省大量内存。

### 8.3 Geo（地理坐标）

存储地理坐标，支持距离计算和范围搜索。底层是 Sorted Set。

```bash
GEOADD locations 121.47 31.23 "Shanghai"
GEOADD locations 116.41 39.91 "Beijing"

GEODIST locations Shanghai Beijing km    # 计算两城市距离（km）
GEOPOS locations Shanghai               # 获取坐标
GEORADIUS locations 121 31 200 km       # 搜索200km范围内的地点
```

### 8.4 Stream（流）

Redis 5.0 引入，是一个**持久化的消息流**，支持：

- 消费者组（Consumer Group）
- 消息确认（ACK）
- 历史消息回溯

比 List 更适合做消息队列中间件，支持多消费者组独立消费。

```bash
XADD events * user_id 1001 action "login"    # 添加事件（* 自动生成ID）
XREAD COUNT 10 STREAMS events 0-0            # 从头读取
XGROUP CREATE events mygroup $ MKSTREAM      # 创建消费者组
XREADGROUP GROUP mygroup consumer1 COUNT 5 STREAMS events >  # 消费
XACK events mygroup 1718380800000-0          # 确认消息处理完成
```

## 9. 如何选择数据类型

| 场景 | 推荐类型 | 理由 |
|------|---------|------|
| 缓存单个值/对象（不需要局部更新） | String | 简单直接 |
| 缓存对象（需要单独更新字段） | Hash | 避免序列化整个对象 |
| 最新列表/时间线 | List | 有序，支持左右端操作 |
| 去重/标签/共同好友 | Set | 自动去重，集合运算 |
| 排行榜/延迟队列 | Sorted Set | 按分数排序，支持范围查询 |
| UV 统计（允许误差） | HyperLogLog | 极省内存 |
| 用户签到/在线状态 | Bitmap | 每个用户一个bit，极省内存 |
| LBS 附近的人 | Geo | 内置距离计算 |
| 消息队列（需要消费确认） | Stream | 持久化，消费者组，ACK |

## 10. 面试回答模板

Redis 有五种核心数据类型：String、List、Hash、Set、Sorted Set，以及 Bitmap、HyperLogLog、Geo、Stream 等特殊类型。

String 是最基础的，适合缓存、计数器、分布式锁；List 有序可重复，适合消息队列和最新动态列表；Hash 适合存储对象，可以单独更新字段；Set 无序不重复，适合去重和集合运算；Sorted Set 有分数排序，适合排行榜和延迟队列。

选择数据类型的核心逻辑是：看你的操作模式——是否需要排序、去重、范围查询、集合运算，然后选对应类型。

## 我应该背下来的最短答案

Redis 五种核心数据类型：String（缓存/计数器/锁）、List（队列/时间线）、Hash（对象存储/购物车）、Set（去重/共同好友）、Sorted Set（排行榜/延迟队列）。特殊类型：HyperLogLog（UV统计）、Bitmap（签到/在线状态）、Geo（地理位置）、Stream（消息流）。
