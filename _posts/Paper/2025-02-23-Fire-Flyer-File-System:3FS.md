---
title: Fire-Flyer File System:3FS
categories: [Paper]
tags: DeepSeek, 3FS 
---

3FS是DeepSeek开源的高性能分布式文件系统，全称Fire-Flyer File System。3FS采用存算分离设计，支持数据强一致以及标准文件接口。
### 1. 整体架构
主要分为四个组件，client 和cluster manager，meta service，storage service，组件都通过 RDMA 网络连接
- cluster manager：元服务和存储服务向集群管理器发送心跳信号，其他服务和client从cluster manager那里获取集群配置和服务状态
- meta service：无状态的，meta信息持久化到DB层；
- client：连接任意的meta service，根据获取到的信息找到相应的storage server，执行相应的IO操作。
- storage service：负责数据的持久化
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-9.png?raw=true" alt="Paper Image" width="300">
</p>

### 2. Meta service
以chunk的粒度进行管理，采用CRAQ的副本策略保证数据一致性，这里的数据放置策略示例如下，假设有6个节点A, B, C, D, E, F，其中每个节点包好1个disk，每个disk上创建5个target（卷），比如节点A包含的target为A1~A5，每个chunk3个副本分布式在不同的node上；
每个chain对应的有版本号，在包含的target发生变化时，cluster manager会修改该版本号（递增）；
在实现中可以实现多种放置策略，也就是可以有多个chain tables，比如一个用于批处理/离线作业，另一个用于在线服务。这两个表由互斥节点上的taregt组成。
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-10.png?raw=true" alt="Paper Image" width="300">
</p>

在创建新文件时，metadata service采用RR的方式选择replicate chain，用于保证数据平衡分布；
当应用程序打开文件时，client会请求metadata service获取文件的数据布局信息。然后，client通过计算获取 chunk IDs和replicate chain，之后向chunk引擎发起IO请求；

### 3. Cluster manager
采用主备保证高可用（etcd等管理），管理集群配置信息，处理成员变更同步给客户端和其他服务；metadata service和storage service向集群管理器发送心跳，集群管理器通过监听信息来检测服务状态

### 4. Storage service
存储节点配置信息
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-11.png?raw=true" alt="Paper Image" width="300">
</p>

在3FS中，存储节点拓扑信息如下：
<p align="center">
  <img src="https://github.com/RuixiangMa/lancer.github.io/blob/main/_posts/Paper/image-12.png?raw=true" alt="Paper Image" width="300">
</p>

chunk engine是Storage service的主要组件，主要包含包含chunk allocator和MetaStore
- chunk allocator:分配和回收chunk
- MetaStore：持久保存分配/回收事件
#### 4.1 chunk allocator
在内存中维护以下三个结构管理chunk
allocated_groups:已分配空间但未指派给chunk
unallocated_groups:未分配磁盘空间
active_groups:维护group的状态<group_id, group_state> 
chunk分配流程：
1. 从active_groups中查找空闲slot
2. 如果active_groups中获取不到空闲slot，则从allocated_groups中取到group
3. 上述情况都不成功，则unallocated_groups中获取，并给unallocated_groups分配disk空间
在分配器中维护两个后台线程：
- allocate_thread：保证active_groups中的大小维护在一定的水位
- compact_thread：回收active_groups中的chunk到allocated_groups
#### 4.2 MetaStore
主要维护三种重要的元数据，MetaStore的信息存储到rocksdb中
chunk_id -> chunk_meta：chunk_meta存储chunk位置，大小等信息
group_id -> group_state：追踪group状态
chunk_pos -> chunk_id：chunk物理位置到chunk id的映射表，用于compaction操作
另外，chunk engine中维护了MetaCache
MetaCache：保存chunk_id -> chunk_info映射表，chunk_info包括chunk_meta等
读操作：返回chunk_info
写操作：
1. 查找MetaCache，检索当前chunk_info信息
2. 调用Allocator::allocate()，分配可写入的chunk位置
3. 执行RMW操作，写到新位置
4. 将new_chunk_info持久化，并且修改之前的映射表信息
另外论文中也提到了基于3FS提供分布式KV存储系统，用于支撑DeepSeek的KV上下文缓存技术，但是目前开源 版本貌似没看到