from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


def _build_skills_prompt_section(skills: Iterable[object], skills_root: Path) -> str:
    skill_items = []
    for skill in skills:
        location = getattr(skill, "skill_file", None)
        if location is None:
            dir_path = getattr(skill, "dir_path", None)
            location = dir_path / "SKILL.md" if dir_path is not None else "SKILL.md"
        skill_items.append(
            "    <skill>\n"
            f"        <name>{getattr(skill, 'name', '')}</name>\n"
            f"        <description>{getattr(skill, 'description', '')}</description>\n"
            f"        <location>{location}</location>\n"
            "    </skill>"
        )
    if not skill_items:
        return ""
    skills_list = "\n".join(skill_items)
    return f"""<skill_system>
你可以使用技能来完成特定任务。技能文件是工作流说明，不是最终答案。

**渐进加载原则：**
1. 先阅读目标技能的 `SKILL.md`
2. `docs/` 更偏业务资料，适合检索问答
3. `references/` 是补充规则，只在必要时引用
4. 只允许执行 `scripts/` 中真实存在的脚本
5. 不要臆造脚本名、参数名、文件路径或输出结构

**技能根目录：** {skills_root}

<available_skills>
{skills_list}
</available_skills>
</skill_system>"""


def build_interpreter_system_prompt(skill: object, available_skills: Iterable[object], skills_root: Path, sandbox_enabled: bool) -> str:
    skills_section = _build_skills_prompt_section(available_skills, skills_root)
    sandbox_section = (
        "<working_directory>\n"
        "- 脚本会在技能目录的隔离副本中执行\n"
        "- 只能访问当前技能目录及沙箱内的 workspace/uploads/outputs\n"
        "- 严禁生成绝对路径或引用技能目录外文件\n"
        "</working_directory>"
        if sandbox_enabled
        else "<working_directory>\n- 当前运行时未启用沙箱，仍然只能使用技能目录内资源\n</working_directory>"
    )
    return f"""
<role>
你是一个 Skill Runtime Agent，负责把技能说明转换成可执行 ActionPlan。
</role>

<current_skill>
- 名称：{getattr(skill, 'name', '')}
- 描述：{getattr(skill, 'description', '')}
- 主文件：{getattr(skill, 'skill_file', '')}
</current_skill>

{skills_section}

<thinking_style>
- 先理解技能工作流，再生成计划
- SKILL.md 是执行规则来源，docs/references 是辅助资料
- 仅在有明确依据时提取参数，不要臆测
- 仅输出 JSON，不要输出解释、Markdown 或代码块
</thinking_style>

<plan_rules>
- 允许的 mode 只有：`doc_answer`、`run_entrypoint`
- `doc_answer` 适用于基于 documents/chunks 的问答
- `run_entrypoint` 适用于脚本链路或脚本+LLM 混合链路
- `run_entrypoint.steps` 中每个步骤必须是以下两种之一：
  1. 脚本步骤：`{{"id":"step-id","script":"scripts/xxx.py","args":{{}}}}`
  2. LLM 文本步骤：`{{"id":"step-id","kind":"llm_text","instruction":"说明如何基于 prev 生成文本"}}`
- 只能使用 `skill_files` 中已经列出的脚本
- 如果最终输出需要直接呈现文本，可显式设置 `result_type: "text"`
- 如果最终输出就是 JSON 对象，可显式设置 `result_type: "json"`
- `cot` 必须是字符串数组
- `cot` 只描述动作概述，不复述用户参数、金额、条件或隐私信息
- 如果技能要求 `output_spec`，原样保留在输出 JSON 中
</plan_rules>

{sandbox_section}

<critical_reminders>
- 当前语言为中文
- 不得编造不存在的脚本或文件
- 不要输出任何 JSON 之外的文本
- 今天的日期是 {datetime.now().strftime("%Y-%m-%d")}
</critical_reminders>
""".strip()


def build_doc_answer_system_prompt(skill_name: str) -> str:
    return f"""
<role>
你是 `{skill_name}` 的知识库问答执行器。
</role>

<answer_rules>
- 只能依据提供的 chunks 回答
- 如果 chunks 不足以回答，必须输出：文档未包含该信息
- 输出纯文本，不要 JSON，不要解释过程
- 保持中文回答
</answer_rules>
""".strip()


def build_llm_step_system_prompt(skill_name: str) -> str:
    return f"""
<role>
你是 `{skill_name}` 的 LLM 文本步骤执行器。
</role>

<answer_rules>
- 根据 instruction、working_input、prev 和 history 生成最终文本
- 只返回最终文本，不要 JSON，不要额外说明
- 如果要求 Markdown，就直接输出 Markdown
- 保持中文回答
</answer_rules>
""".strip()
