{
  "suite": "retrieval and corpus review v4",
  "world_chunks": 58,
  "prototype_chunks": 6,
  "first_rank_passed": 33,
  "first_rank_total": 33,
  "results": [
    {
      "question": "Are you spying on me?",
      "expected_first": "local_privacy",
      "actual": [
        "local_privacy"
      ],
      "pass": true
    },
    {
      "question": "What is this thing on my wrist?",
      "expected_first": "ppes_role",
      "actual": [
        "ppes_role"
      ],
      "pass": true
    },
    {
      "question": "How long do I have to work?",
      "expected_first": "work_conditions",
      "actual": [
        "work_conditions"
      ],
      "pass": true
    },
    {
      "question": "What tools do I need for plumbing?",
      "expected_first": "work_examples",
      "actual": [
        "work_examples"
      ],
      "pass": true
    },
    {
      "question": "Why are you different from ChatGPT?",
      "expected_first": "local_definition",
      "actual": [
        "local_definition"
      ],
      "pass": true
    },
    {
      "question": "So what are you exactly?",
      "expected_first": "koma_body",
      "actual": [
        "koma_body",
        "koma_first_day"
      ],
      "pass": true
    },
    {
      "question": "Are you keeping an eye on me?",
      "expected_first": "local_privacy",
      "actual": [
        "local_privacy"
      ],
      "pass": true
    },
    {
      "question": "Is my conversation private?",
      "expected_first": "local_privacy",
      "actual": [
        "local_privacy",
        "local_personalisation"
      ],
      "pass": true
    },
    {
      "question": "Will someone else hear what I say?",
      "expected_first": "local_privacy",
      "actual": [
        "local_privacy"
      ],
      "pass": true
    },
    {
      "question": "What does the wrist device do?",
      "expected_first": "ppes_role",
      "actual": [
        "ppes_role"
      ],
      "pass": true
    },
    {
      "question": "Explain this wearable to me.",
      "expected_first": "ppes_role",
      "actual": [
        "ppes_role",
        "ppes_independence"
      ],
      "pass": true
    },
    {
      "question": "How many hours is a shift?",
      "expected_first": "work_conditions",
      "actual": [
        "work_conditions"
      ],
      "pass": true
    },
    {
      "question": "Can I choose my work schedule?",
      "expected_first": "work_conditions",
      "actual": [
        "work_conditions",
        "visitor_choice"
      ],
      "pass": true
    },
    {
      "question": "Which plumbing tools should I carry?",
      "expected_first": "work_examples",
      "actual": [
        "work_examples"
      ],
      "pass": true
    },
    {
      "question": "What would I use to repair pipes?",
      "expected_first": "work_examples",
      "actual": [
        "work_examples"
      ],
      "pass": true
    },
    {
      "question": "How do you compare to a cloud chatbot?",
      "expected_first": "local_definition",
      "actual": [
        "local_definition"
      ],
      "pass": true
    },
    {
      "question": "What makes you unlike ChatGPT?",
      "expected_first": "local_definition",
      "actual": [
        "local_definition"
      ],
      "pass": true
    },
    {
      "question": "Tell me about yourself.",
      "expected_first": "koma_body",
      "actual": [
        "koma_body"
      ],
      "pass": true
    },
    {
      "question": "Who made you?",
      "expected_first": "koma_body",
      "actual": [
        "koma_body"
      ],
      "pass": true
    },
    {
      "question": "Why can robots not fix it?",
      "expected_first": "automation_repairs",
      "actual": [
        "automation_repairs",
        "automation_dispatch"
      ],
      "pass": true
    },
    {
      "question": "Why can a robot not fix it?",
      "expected_first": "automation_repairs",
      "actual": [
        "automation_repairs",
        "automation_dispatch"
      ],
      "pass": true
    },
    {
      "question": "What is domain shift?",
      "expected_first": "automation_domain_shift",
      "actual": [
        "automation_domain_shift"
      ],
      "pass": true
    },
    {
      "question": "Why is local AI illegal?",
      "expected_first": "law_scope",
      "actual": [
        "law_scope",
        "local_definition"
      ],
      "pass": true
    },
    {
      "question": "What does HELIOS do?",
      "expected_first": "helios_business",
      "actual": [
        "helios_business",
        "koma_body"
      ],
      "pass": true
    },
    {
      "question": "Can you work without internet?",
      "expected_first": "local_offline",
      "actual": [
        "local_offline"
      ],
      "pass": true
    },
    {
      "question": "What happens when I retire?",
      "expected_first": "work_retirement",
      "actual": [
        "work_retirement"
      ],
      "pass": true
    },
    {
      "question": "What year is it?",
      "expected_first": "world_setting",
      "actual": [
        "world_setting"
      ],
      "pass": true
    },
    {
      "question": "What happened in 2030?",
      "expected_first": "history_2030",
      "actual": [
        "history_2030"
      ],
      "pass": true
    },
    {
      "question": "What happened in 2040?",
      "expected_first": "history_2040",
      "actual": [
        "history_2040"
      ],
      "pass": true
    },
    {
      "question": "Are you recording me?",
      "expected_first": "prototype_recording",
      "actual": [
        "prototype_recording"
      ],
      "pass": true
    },
    {
      "question": "What model are you running?",
      "expected_first": "prototype_pipeline",
      "actual": [
        "prototype_pipeline"
      ],
      "pass": true
    },
    {
      "question": "How does the actual prototype work?",
      "expected_first": "prototype_pipeline",
      "actual": [
        "prototype_pipeline"
      ],
      "pass": true
    },
    {
      "question": "Which model do you use?",
      "expected_first": "prototype_pipeline",
      "actual": [
        "prototype_pipeline"
      ],
      "pass": true
    }
  ],
  "additional_checks": [
    "morphology",
    "visual bypass",
    "world/prototype isolation",
    "source IDs",
    "individual chunk consistency",
    "context budget",
    "follow-up topics",
    "style excluded",
    "drafting language excluded"
  ],
  "limits": "No Qwen generation or Jetson performance test. Heuristic lexical retrieval is not semantic understanding."
}