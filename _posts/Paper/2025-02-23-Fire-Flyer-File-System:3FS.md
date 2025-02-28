---
title: Fire-Flyer File System:3FS
categories: [Paper]
tags: DeepSeek, 3FS
---

*3FS*是*DeepSeek*开源的高性能分布式文件系统，全称*Fire-Flyer File System*。*3FS*采用存算分离设计，支持数据强一致以及标准文件接口。
### 1. 整体架构
主要分为四个组件，*client* 和*cluster manager*，*meta service*，*storage service*，组件都通过 *RDMA* 网络连接
- *cluster manager*：管理集群配置，采用主备保证高可用（*etcd*等管理）；
- *meta service*：无状态，多个*meta service*支持高扩展，*meta*信息持久化到*DB*层（*FoundationDB*）；
- *client*：基于*Fuse*的客户端，支持连接任意的*meta service*，根据获取到的信息找到相应的*storage server*，执行相应的*IO*操作。
- *storage service*：负责数据的持久化
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-9.png?raw=true" alt="Paper Image" width="300">
</p>

### 2. Meta service
以*chunk*的粒度进行管理，采用*CRAQ*的副本策略保证数据一致性;
数据放置策略示例如下，假设有6个节点*A*, *B*, *C*, *D*, *E*, *F*，其中每个节点只有1个*disk*，每个*disk*上创建5个*target*；比如节点*A*包含的*target*为*A1~A5*，每个*chunk*的3个副本会分布在不同的*target*上，对应一个*chain*；每个*chain*对应的有版本号，当包含的*target*发生变化时，*cluster manager*会修改该版本号（递增）；

在实现中可以实现多种放置策略，也就是可以有多个*chain tables*，比如一个用于批处理/离线作业，另一个用于在线服务。这两个*chain table*由互斥节点上的*target*组成。
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-10.png?raw=true" alt="Paper Image" width="300">
</p>

在创建新文件时，*metadata service*采用*RR*的方式选择*replicate chain*，用于保证数据平衡分布；
当应用程序打开文件时，*client*会请求*metadata service*获取文件的数据布局信息。然后，*client*通过计算获取 *chunk IDs*和*replicate chain*，之后向*chunk*引擎发起*IO*请求；

### 3. Cluster manager
管理集群配置信息，处理成员变更同步给客户端和其他服务；在运行时，*metadata service*和*storage service*向*Cluster manager*发送心跳，集群管理器通过监听信息来检测服务状态

### 4. Storage service
存储节点配置信息
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-11.png?raw=true" alt="Paper Image" width="300">
</p>

在*3FS*中，存储节点拓扑信息如下：
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-12.png?raw=true" alt="Paper Image" width="300">
</p>

*chunk engine*是*Storage service*的主要组件，主要包含包含*chunk allocator*和*MetaStore*
- *chunk allocator*:分配和回收*chunk*
- *MetaStore*：持久保存分配/回收事件
#### 4.1 chunk allocator
在内存中维护以下三个结构管理*chunk*

*allocated_groups*:已分配空间但未指派给*chunk*

*unallocated_groups*:未分配磁盘空间

*active_groups*:维护*group*的状态<*group_id*, *group_state*>

*chunk*分配流程：
1. 从*active_groups*中查找空闲*slot*
2. 如果*active_groups*中获取不到空闲*slot*，则从*allocated_groups*中取到*group*
3. 上述情况都不成功，则*unallocated_groups*中获取，并给*unallocated_groups*分配*disk*空间

在分配器中维护两个后台线程：
- *allocate_thread*：保证*active_groups*中的大小维护在一定的水位
- *compact_thread*：回收*active_groups*中的*chunk*到*allocated_groups*
#### 4.2 MetaStore
主要维护三种重要的元数据，*MetaStore*的信息存储到*rocksdb*中

*chunk_id* -> *chunk_meta*：*chunk_meta*存储*chunk*位置，大小等信息

*group_id* -> *group_state*：追踪*group*状态

*chunk_pos* -> *chunk_id*：*chunk*物理位置到*chunk id*的映射表，用于*compaction*操作

另外，*chunk engine*中维护了*MetaCache*
*MetaCache*：保存*chunk_id* -> *chunk_info*映射表，*chunk_info*包括*chunk_meta*等

读操作：返回*chunk_info*

写操作：
1. 查找*MetaCache*，检索当前*chunk_info*信息
2. 调用*Allocator::allocate()*，分配可写入的*chunk*位置
3. 执行*RMW*操作，写到新位置
4. 将*new_chunk_info*持久化，并且修改之前的映射表信息
另外论文中也提到了基于*3FS*提供分布式*KV*存储系统，用于支撑*DeepSeek*的*KV*上下文缓存技术，但是目前开源 版本貌似没看到