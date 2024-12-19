Q: TuGraph 的边是否支持索引？
A: TuGraph 在引擎层支持边索引，可通过存储过程使用。Cypher的边索引功能正在开发支持中。

Q: TuGraph 单机的QPS是多少？
A: 不同数据规模，不同查询操作的QPS差异较大，比如LDBC SNB典型图操作超过1.2万，参考测试结果：https://www.tugraph.org/blog?id=0

Q: 可视化文件 build 后如何更新到 tugraph 服务？
A: 可视化文件打包后，需要进行以下操作进行替换。

- 登录 tugraph 服务所在的服务或 docker 容器内。
- 通过 lgraph_server --help 查看服务启动的配置文件所在目录。通常情况：`/usr/local/etc/lgraph.json`
- 查看 `/usr/local/etc/lgraph.json`文件中 web 的配置目录。通常情况：`/usr/local/share/lgraph/resource`
- 将可视化打包后生成的文件夹中的内容全部替换到配置目录下 `/usr/local/share/lgraph/resource`
- 重新启动 tugraph 服务

Q：client 目前有哪些编程语言，是否支持 node js？
A：目前主要支持的编程语言有 c++,python,java；目前不支持 node js。使用 node 作为主要开发语言的用户，可以使用 tugraph 提供的 restful api 来调用。建议使用 Cypher 来封装调用接口。后续版本 restful api 将不再进行更新维护，只会保留登录、登出、刷新 token、cypher 调用这几个常见的 api。

Q：python client 是否支持 pip install？client 在哪里进行引用？
A：目前 python client 不支持 pip 进行安装。client 在目录https://github.com/TuGraph-db/tugraph-db/tree/master/src/client。

Q：TuGraph c++ SDK如何获取或构建？
A：我们有一个c++版本的旧版client，后续不再主动更新。当前我们已经升级我们的client-server协议支持bolt，但缺少c++的bolt client，如感兴趣我们欢迎和我们一起共建。
旧版本C++客户端实例在：https://github.com/TuGraph-family/tugraph-db/tree/master/test/test_rpc_client/cpp/CppClientTest

Q：Rust如何来访问Tugraph Server呢？
A：现在TuGraph的客户端统一都走bolt，目前是支持rust的，https://github.com/TuGraph-family/tugraph-db/blob/master/demo/Bolt/rust_example.rs

Q：github下载的源码压缩包里没有包含deps项目的源码，导致编译时执行deps/build_deps.sh，报找不到package.json的错误。
A：目前源码包里不包含deps源码，后续会更新。先使用git拉取的方式拉取全部源码。并推荐使用tugraph编译镜像进行源码编译，从而减少编译过程中环境配置等问题。

Q：如何通过 npm run dev，连接已有的 tugraph 服务？
A：启动之前，需要修改文件`.env.development`中的'VUE_APP_REQUESTURL'的配置项。然后在通过`npm run dev`进行启动。
示例：
`NODE_ENV = development VUE_APP_TITLE = TuGraph(dev) VUE_APP_REQUESTURL = http://localhost:7070/`

Q: TuGraph 可以对接那些常用数据库？
A: TuGraph通过DataX可以实现大部分主流数据库的导入导出，支持的数据库包括MySQL、Oracle、Hive 等。具体参考https://github.com/TuGraph-db/DataX

Q：如何进行数据导入？
A：数据导入可见官网文档：https://www.tugraph.org/doc?version=V3.3.0&id=10000000000658643

Q：读取oracle数据报错
"error_message":"Error parsing file memory_file_stream\n\tError occurred at offset 0, exception detail:\n\tjson reading failed, error msg : std::bad_cast\n>Error line...."，如何解决？
A：看起来像在处理数据的时候遇到特使符号导致报错的，建议用相对较小的表以及数据可以尝试测一下

Q：value pack时的null array的具体含义是？
A：标记这个schema的field是否为空，如果一个schema的field为空，并且插入的数据里对应的列是空的，在packed的时候就不占内存

Q：怎么把存储于tugraph的某些指定的点/边类型的全量数据，导出到csv文件中？
A：使用neo4j driver 连接tugraph，直接发送cypher 语句 "match (n) return n" 就可以了。结果是流式返回的，不管多少数据都可以读出来，不会引发内存oom

Q：如何加载存储过程或算法包？
A：加载方式有两种：

- 第一种：通过可视化页面的插件模块，通过交互操作完成加载。
- 第二种：通过 cypher 语句实现存储过程的加载。

```
CALL db.plugin.loadPlugin(plugin_type::STRING,plugin_name::STRING,plugin_content::STRING,code_type::STRING,plugin_description::STRING,read_only::BOOLEAN) :: (::VOID)
```

Q：如何调用或执行存储过程？
A：可以使用 cypher 进行存储过程的执行或调用。

```
CALL db.plugin.callPlugin(plugin_type::STRING,plugin_name::STRING,param::STRING,timeout::DOUBLE,in_process::BOOLEAN) :: (success::BOOLEAN,result::STRING)
```

Q:开源内置的算法包在哪里？
A：代码地址https://github.com/TuGraph-db/tugraph-db/tree/master/plugins

Q：如何使用 docker 镜像安装？
A：

- 确认本地是否有 docker 环境，可使用`docker -v`进行验证。如果没有请安装 docker，安装方式见 docker 官网文档https://docs.docker.com/install/ 。
- 下载 docker 镜像，下载方式可使用`docker pull tugraph/tugraph-runtime-centos7`，也可以在官网下载页面进行下载https://www.tugraph.org/download**[注：下载的文件是\*.tar.gz 的压缩包，不用解压]**。
- 如果使用 docker pull 下载的镜像则不用导入镜像。如果使用官网下载的压缩包，则要使用`docker load -i ./tugraph_x.y.z.tar`**[注：x.y.z 是版本号的代替符，具体数值根据自己下载的版本进行改写]**
- 启动 docker 容器`docker run -d -p 7070:7070 -p 9090:9090 --name tugraph_demo tugraph/tugraph-runtime-centos7 lgraph_server`**[注：具体的镜像名称 tugraph/tugraph-runtime-centos7 要以本地实际镜像名称为准，可用过 docker images 命令查看]**

Q：rpm 包和 deb 包安装后，启动 lgraph_server 服务。提示缺少'liblgraph.so'报错？
A：此问题主要是环境变量导致，需要配置环境量。

Q：如何打开审计功能
A：需要打开enable_audit_log开关，具体参考" https://github.com/TuGraph-family/tugraph-db/blob/master/docs/zh-CN/source/10.permission/5.log.md " 中的审计日志

Q：产品是否支持麒麟操作系统？只有企业版支持么？
A：开源和企业版都支持

Q：是否支持不定长边的条件查询？
示例：

```
 MATCH p=(v)-[e:acted_in|:rate*1..3]-(v2) WHERE id(v) IN [3937] AND e.stars = 3 RETURN p LIMIT 100
```

A：目前还不支持不定长边的过滤查询。目前的代替方案只能是分开写。上面的示例，就需要从 1 跳到 3 跳都写一遍。

Q：如何查询最短路径，shortestPath 函数如何使用？
A：使用示例如下（示例图谱：MovieDemo）

```
 MATCH (n1 {name:'Corin Redgrave'}),(n2 {name:'Liam Neeson'})
     CALL algo.allShortestPaths(n1,n2) YIELD nodeIds,relationshipIds,cost
         RETURN nodeIds,relationshipIds,cost
```

Q：查询语句 Where 后使用 and 进行拼接查询速度较慢，语句应如何优化改进？
示例：

```
 MATCH (n1),(n2) CALL algo.allShortestPaths(n1,n2)
     YIELD nodeIds,relationshipIds,cost
         WHERE id(n1) IN [0] AND id(n2) IN [3938]
             RETURN nodeIds,relationshipIds,cost
```

A：目前 cypher 查询引擎正在优化中。现阶段语句改写可以通过 with 向下传递进行优化。

Q：如何查询任意跳的边？
A：使用`*..`
示例：

```
 MATCH p=(a)-[*..]-(b) WHERE id(a) IN [3] AND id(b) IN [19] RETURN p
```

Q：MERGE语法报错（"BadRequest CypherException: Function not implemented yet: ExtractNodePattern at :166"），查询语句（MERGE (n1:domain {name:'root',id: 'root', tag: 'root'})）
A：现在对于有多个属性的这种Merge语法还没完全支持，可以参考这几种用法：

1. MERGE (n:Person {name:'Zhugeliang'}) ON CREATE SET n.gender=1,n.birthyear=181 RETURN n.name
2. MERGE (n:Person {name:'Liubei'}) ON MATCH SET n.birthyear=2010 RETURN n.birthyear

Q：如何获取每个Cypher请求的耗时?
A：TuGraph的cli可以看到时间：
https://github.com/TuGraph-family/tugraph-db/blob/master/docs/zh-CN/source/6.utility-tools/6.tugraph-cli.md
如果是白屏前端，可以看接口调用的端到端时间

Q：多条件场景下，查询报错，语法如下：MATCH (tom:Person{name:"Andres",title:"Developer"}) RETURN tom
需要修改成：MATCH (tom:Person{name:"Andres"}) RETURN tom才能正确执行，麻烦看下是否不支持多条件查询，或者是否有其他的替代查询方式
A：可以尝试这样查询
MATCH (tom:Person) WHERE tom.name = 'Andres' AND tom.title = 'Developer' RETURN tom

Q：报错"User has reached the maximum number of tokens"后，怎么做？
A：这表明当前账号Token数量已达上限10000个。解决方法如下，任选其一：

1. 登出不使用的Token。
2. 重新启动TuGraph服务，会清空所有Token。
3. Token有效期默认为24小时，24小时后会自动失效并删除。

1. RETURN datetime("2022-10-18 00:00:00") AS a (Success)
2. RETURN datetime("2022-10-18") AS a

- failed to parse string 2022-10-18 into DateTime

1. RETURN date("2022-10-18") AS a

- unregistered function: date

Q：Did I make a mistake on Cypher syntax, or please support the date() function. Thanks.

A：The `date()` function is not implemented yet, we will implement it later

Q：查询语句 Where 后使用 and 进行拼接查询速度较慢，语句应如何优化改进？
示例：

```
 MATCH (n1),(n2) CALL algo.allShortestPaths(n1,n2)
     YIELD nodeIds,relationshipIds,cost
         WHERE id(n1) IN [0] AND id(n2) IN [3938]
             RETURN nodeIds,relationshipIds,cost
```

A：目前 cypher 查询引擎正在优化中。现阶段语句改写可以通过 with 向下传递进行优化。
示例：

```
 MATCH (n1) where id(n1) in [0] with n1
 MATCH (n2) where id(n2) in [3938] with n1, n2
     CALL algo.allShortestPaths(n1,n2) YIELD nodeIds,relationshipIds,cost
         RETURN nodeIds,relationshipIds,cost
```

Q：如何查询任意跳的边？
A：使用`*..`
示例：

```
 MATCH p=(a)-[*..]-(b) WHERE id(a) IN [3] AND id(b) IN [19] RETURN p
```

Q：MERGE语法报错（"BadRequest CypherException: Function not implemented yet: ExtractNodePattern at :166"），查询语句（MERGE (n1:domain {name:'root',id: 'root', tag: 'root'})）
A：现在对于有多个属性的这种Merge语法还没完全支持，可以参考这几种用法：

1. MERGE (n:Person {name:'Zhugeliang'}) ON CREATE SET n.gender=1,n.birthyear=181 RETURN n.name
2. MERGE (n:Person {name:'Liubei'}) ON MATCH SET n.birthyear=2010 RETURN n.birthyear

Q：如何获取每个Cypher请求的耗时?
A：TuGraph的cli可以看到时间：
https://github.com/TuGraph-family/tugraph-db/blob/master/docs/zh-CN/source/6.utility-tools/6.tugraph-cli.md
如果是白屏前端，可以看接口调用的端到端时间

Q：多条件场景下，查询报错，语法如下：MATCH (tom:Person{name:"Andres",title:"Developer"}) RETURN tom
需要修改成：MATCH (tom:Person{name:"Andres"}) RETURN tom才能正确执行，麻烦看下是否不支持多条件查询，或者是否有其他的替代查询方式
A：可以尝试这样查询
MATCH (tom:Person) WHERE tom.name = 'Andres' AND tom.title = 'Developer' RETURN tom

Q：查询语句 Where 后使用 and 进行拼接查询速度较慢，语句应如何优化改进？
示例：

```
 MATCH (n1),(n2) CALL algo.allShortestPaths(n1,n2)
     YIELD nodeIds,relationshipIds,cost
         WHERE id(n1) IN [0] AND id(n2) IN [3938]
             RETURN nodeIds,relationshipIds,cost
```

A：目前 cypher 查询引擎正在优化中。现阶段语句改写可以通过 with 向下传递进行优化。
示例：

```
 MATCH (n1) where id(n1) in [0] with n1
 MATCH (n2) where id(n2) in [3938] with n1, n2
     CALL algo.allShortestPaths(n1,n2) YIELD nodeIds,relationshipIds,cost
         RETURN nodeIds,relationshipIds,cost
```

Q：

```
terminate called without an active exception
Aborted (core dumped)
```

A：To support aarch64 with non-4K pagesize:

- recompile and reinstall jemalloc libraries. Reason: jemalloc detects the page size at compile time for performance. (Double check the -fPIC options)
- rebuilt the tugraph binaries with the new jemalloc binaries. `cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_ON_AARCH64=ON`

we have an amazing guide for compile TuGraph in mac

https://zhuanlan.zhihu.com/p/561139698

Q：rpm 包和 deb 包安装后，启动 lgraph_server 服务。提示缺少'liblgraph.so'报错？
A：此问题主要是环境变量导致，需要配置环境量。
示例：

```
export LD_LIBRARY_PATH=/usr/local/lib64
```

Q：如何打开审计功能
A：需要打开enable_audit_log开关，具体参考" https://github.com/TuGraph-family/tugraph-db/blob/master/docs/zh-CN/source/10.permission/5.log.md " 中的审计日志

Q：产品是否支持麒麟操作系统？只有企业版支持么？
A：开源和企业版都支持

Q:开源内置的算法包在哪里？
A：代码地址https://github.com/TuGraph-db/tugraph-db/tree/master/plugins

Q：如何加载存储过程或算法包？
A：加载方式有两种：

- 第一种：通过可视化页面的插件模块，通过交互操作完成加载。
- 第二种：通过 cypher 语句实现存储过程的加载。

```
CALL db.plugin.loadPlugin(plugin_type::STRING,plugin_name::STRING,plugin_content::STRING,code_type::STRING,plugin_description::STRING,read_only::BOOLEAN) :: (::VOID)
```

Q：读取oracle数据报错
"error_message":"Error parsing file memory_file_stream\n\tError occurred at offset 0, exception detail:\n\tjson reading failed, error msg : std::bad_cast\n>Error line...."，如何解决？
A：看起来像在处理数据的时候遇到特使符号导致报错的，建议用相对较小的表以及数据可以尝试测一下

Q：python client 是否支持 pip install？client 在哪里进行引用？
A：目前 python client 不支持 pip 进行安装。client 在目录https://github.com/TuGraph-db/tugraph-db/tree/master/src/client。

Q：TuGraph c++ SDK如何获取或构建？
A：我们有一个c++版本的旧版client，后续不再主动更新。当前我们已经升级我们的client-server协议支持bolt，但缺少c++的bolt client，如感兴趣我们欢迎和我们一起共建。
旧版本C++客户端实例在：https://github.com/TuGraph-family/tugraph-db/tree/master/test/test_rpc_client/cpp/CppClientTest

Q：Rust如何来访问Tugraph Server呢？
A：现在TuGraph的客户端统一都走bolt，目前是支持rust的，https://github.com/TuGraph-family/tugraph-db/blob/master/demo/Bolt/rust_example.rs

Q：github下载的源码压缩包里没有包含deps项目的源码，导致编译时执行deps/build_deps.sh，报找不到package.json的错误。
A：目前源码包里不包含deps源码，后续会更新。先使用git拉取的方式拉取全部源码。并推荐使用tugraph编译镜像进行源码编译，从而减少编译过程中环境配置等问题。

Q：client 目前有哪些编程语言，是否支持 node js？
A：目前主要支持的编程语言有 c++,python,java；目前不支持 node js。使用 node 作为主要开发语言的用户，可以使用 tugraph 提供的 restful api 来调用。建议使用 Cypher 来封装调用接口。后续版本 restful api 将不再进行更新维护，只会保留登录、登出、刷新 token、cypher 调用这几个常见的 api。

Q：HA集群的snapshot何时删除？
A：会在生成下一个snapshot时删除上一个snapshot，具体时长可以使用snapshot_interval_s配置。

Q：请问一下Tugraph支持向量检索or混合检索（向量+关键词）吗？
A：暂时还不支持向量的存储和检索。

Q: 能否像neo4j一样在同一个节点上添加多个标签？
A: TuGraph目前还没有支持多标签。你可以尝试添加标签属性，用来表示原本你想用作标签的信息，或者尝试每个标签创建一个类型的点模型。

Q: 是否支持自定义和调用高性能C++计算函数？
A: 都是用存储过程，由于是在server端执行，所以性能没问题，SNB的测试就是这么跑的

Q：4.3.2版本的镜像，启动成功了，进入容器后没有 setup.sh 脚本是目录换了吗？
A：从TuGraph 4.3.2版本及其后续版本开始，设计上已经简化了启动流程。当你运行TuGraph的Docker镜像之后，图数据库服务和Web浏览器（Browser）会自动启动，无需再手动执行额外的启动脚本如setup.sh。这样的改进旨在为用户提供更加即开即用的体验，减少了部署和初始化的复杂度。

Q：启动TuGraph的时候报这个错误：0x00007f7e5f272900 FATAL include/fma-common/binary_buffer.h:289] CHECK(gpos_ + size <= ppos_) failedreading beyond the array: required size=4, actual size=2
A：因版本升级幅度较大而导致的兼容性问题，建议先清除原有的数据目录（默认是/var/lib/lgraph/data），然后再进行重试。请注意，此操作将永久删除所有存储在该路径下的数据，请确保已做好相应数据备份。

Q：TuGraph版本更新之后，原库的数据会丢吗？
A：按照官方手册启动容器的时候会把数据目录映射到物理机的/root/tugraph/data目录，只要没有删除物理机/root/tugraph/data目录数据就不会丢。

Q：TuGraph Docker单机镜像环境中，如何获取每条cypher查询语句的耗时
A：使用cli调用cypher的时候可以看到时间 https://tugraph-db.readthedocs.io/zh-cn/latest/5.developer-manual/4.client-tools/5.tugraph-cli.html

TuGraph快速部署建议：建议在新手快速上手的时候，可以考虑多平台镜像快速使用。例如考虑一个TuGraph的helm仓库，支持helm 一行命令直接部署，开箱即用。目前有很多用户都遇到安装过程中遇到依赖缺失等部署问题，希望能部署运行依赖也打包进去。

Q：是否支持自定义和调用高性能C++计算函数？
A：都是用存储过程，由于是在server端执行，所以性能没问题，SNB的测试就是这么跑的。

Q：现在3.4版本是否支持批量更新点和边，如果不支持？后续是否有计划支持
A：使用更加“标准”的Cypher进行批量更新是一个重要的增强接口，我们会将其纳入计划，也可以用graph_import工具，或手写存储过程。

Q：load plugin报错，(400, 'Failed to compile plugin. \nStdout:----\n\nStderr:----\n/var/lib/lgraph/data/5FFB2BF9573C8B03590AEB4ACCDD3F17/*cpp_plugin*/_fma_bfs1540383426_1675404216089603349.tmp/_fma_bfs.cpp:16:10: fatal error: ./algo.h: No such file or directory\n #include "./algo.h"\n ^~~~~~~~~~\ncompilation terminated.\n')
A：这个看报错，就是没有algo.h，改到同一个文件里，或者整一个zip包放上去

Q：python client调用图算法，需要先编译么
A：图算法属于存储过程，无论是什么客户端，都需要先加载（上传在服务端编译），再调用运行。