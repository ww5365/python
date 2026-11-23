## cursor安装和使用

### cursor安装
下载cursor：https://cursor.com/cn/home


### 使用

#### 自动代码生成和调试

* ctrl + L ： 打开和关闭智能对话窗口，通过这个对话窗，可以询问怎么编程，代码的分析

* 问题分析：让ai添加日志，定位问题，把输出日志可以再给cursor，让它定位问题。类似调bug过程。

* ctrl + K ： 局部代码生成，光标的停在需要优化的位置，根据用户描述，进行代码生成

* 选中代码优化： 选中待优化的代码 -> 点击：add to chat -> 按照用户的query进行优化

* 错误诊断和修复： 找到报错的代码片段 -> fix in chat (ctrl + shift + D)-> chat中自动生成：提示词 修复代码

#### 文档

* api文档生成： md文件中写prompt，把源文件路径，规则，写清楚  -> @md文件  -> query：生成api文档
* 本地文件，博客等网站，代码库： cursor可以参考来进行代码编写： **这个功能很重要**
  cursor setting -> indexing&Docs -> Docs(add Doc)
  加下对这个库中代码的参考：https://github.com/NVIDIA/recsys-examples/tree/main/corelib/dynamicemb

#### @的使用

手动**添加上下文**的方式	用法示例	目的
- @Files （引用文件）	“参照 **@src/utils/helper.ts** 的写法，创建一个新的日志工具函数。”	让AI精确读取某个或多个文件，避免它“猜”错文件。
- @Code / @Symbols （引用代码/符号）	“帮我解释一下 ** @calculateTotal **这个函数的逻辑。”	当你只想关注某个特定的函数或类，而不是整个文件时使用，更加精准。
- @Folders （引用文件夹）	“检查一下 **@components** 文件夹下所有组件的 props 命名是否规范。”	一次性为整个目录下的文件提供上下文，适合做全局性的审查或重构。
- @Web （联网搜索）	**“@Web 最新版的 React 19 有哪些新特性？**”	让 AI 突破知识截止日期，联网查找最新信息。
- @Git （引用Git信息）	“**根据 @git 中暂存的更改，帮我写一份提交信息。**”	让 AI 分析你的代码变更记录，辅助代码审查或生成提交信息。



git 自动提交工具？



## claude code 


### claude的安装

####  安装过程
1. Node.js 22+  :  先安装nodejs 
    >> 官方安装包 (.msi): 访问 Node.js 官网，下载 LTS 版本的 .msi 安装包，双击运行，按提示完成安装即可
    >> 版本：node -v  npm -v
2. 安装claude code : npm install -g @anthropic-ai/claude-code
3. 验证：claude --version
4. 更新：npm update -g @anthropic-ai/claude-code   

#### 配置

1. 大模型：参考：https://gitcode.com/org/openCsiTool/discussions/1
2. 配置：C:\Users\Administrator\.claude\settings.json  配置自定义的模型
```
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "API-key",
    "ANTHROPIC_BASE_URL": "IP或网址",
    "ANTHROPIC_CODE_NONESSENTIAL_TRAFFIC": "1",
    "ANTHROPIC_MODEL": "GLM-5.2",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-5.1",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Qwen3.7-Plus",
    "CLAUDE_CODE_SUBAGENT_MODEL": "Qwen3.7-Max"
  },
  "model": "GLM-5.2",
  "theme": "dark",
  "respectGitignore": false,
  "effortLevel": "medium"
}

```

### clude 使用


#### 基本使用
* build 和 plan 模型有什么区别？
build是主代理模式，可以调用工具，进行日常开发，修改代码。
plan模型进行代码分析，提出建议或修改计划，不希望改动代码。

* @general 通用子代理，可以修改代码，几乎全部工具权限(除todo外),
适合场景：比较复杂的问题，需要多步，或多个独立的工作单元，共同完成。
模式1： @general 根据task.md的描述，帮忙我完成积分系统的开发。系统会开一个子代理，使用多步来完成任务； 
模式2：想开多个子代理来完成任务，可以多次调用@general
第一次：@general 根据task.md的描述，帮忙我完成积分系统的前端开发；
第二次紧接输入@general 根据task.md的描述，帮忙我完成积分系统的后端开发；



#### Skills 和 MCP

Skills：是封装好的提示词 / 标准化工作流，让 Claude 变成特定领域的「专业人士」，本质是让 AI 「更懂怎么干」


MCP 服务器：是真正的工具能力，能让 Claude 访问本地文件、浏览器、外部 API、第三方工具，本质是让 AI 「真的能去干」


##### skill 查找安装和使用 


###### npx skils 安装
```text
# 1. 搜索社区技能（关键词匹配）
npx skills find <关键词>

# 2. 安装技能（-y 跳过确认，-g 全局安装，必加！）
npx skills add <owner/repo@skill> -y -g


安装find-skills 用来找skill的skill
npx skills add https://github.com/vercel-labs/skills --skill find-skills

可以用find命令来找skills了
npx skills find "skills的关键词"


# 3. 查看已安装的全部技能
npx skills list -g

也可以通过目录查看claude下安装的skills:
C:\Users\Administrator\.claude\skills


# 4. 检查技能更新
npx skills check

# 5. 更新所有已安装技能
npx skills update

```

典型的使用过程：

```text
1. 查找skill：skill-creator  创建skills的skill
npx skills find "skill-creator"

anthropics/skills@skill-creator 356.7K installs
└ https://skills.sh/anthropics/skills/skill-creator

2. 安装这个skills

npx skills add https://github.com/anthropics/skills --skill skill-creator

3. 使用这个skills

/skills  claude中能看到安装的skills

输入：生成一个将markdown转换为pdf格式skill

结果：C:\Users\Administrator\.claude\skills\md-to-pdf  这个目录下生成skills


```








