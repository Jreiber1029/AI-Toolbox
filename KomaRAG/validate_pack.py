"""Retrieval/routing regressions; does not test Qwen generation or hardware."""
import hashlib
import json
from pathlib import Path
import re
from koma_demo import KomaKnowledge, build_prompt, definition_subject, tokens, route_query

ROOT=Path(__file__).resolve().parent
kb=KomaKnowledge()
checks=[
 ('Tell me about the future we live in','world_setting'),
 ('What is the world like in 2070?','world_setting'),
 ('What is life like here?','world_setting'),
 ('What kind of world is this?','world_setting'),
 ('What do TradesMasters do?','automation_repairs'),
 ('What does a TradesMaster do?','automation_repairs'),
 ('What is a TradesMaster?','automation_repairs'),
 ('What is my job?','work_examples'),
 ('Are you spying on me?','local_privacy'),
 ('What is this thing on my chest?','q4_role'),
 ('How long do I have to work?','work_conditions'),
 ('What tools do I need for plumbing?','work_examples'),
 ('Why are you different from ChatGPT?','local_definition'),
 ('So what are you exactly?','koma_body'),
 ('Are you keeping an eye on me?','local_privacy'),
 ('Is my conversation private?','local_privacy'),
 ('Will someone else hear what I say?','local_privacy'),
 ('What does the chest strap device do?','q4_role'),
 ('Explain this wearable to me.','q4_role'),
 ('How many hours is a shift?','work_conditions'),
 ('Can I choose my work schedule?','work_conditions'),
 ('Which plumbing tools should I carry?','work_examples'),
 ('What would I use to repair pipes?','work_examples'),
 ('How do you compare to a cloud chatbot?','local_definition'),
 ('What makes you unlike ChatGPT?','local_definition'),
 ('Tell me about yourself.','koma_body'),
 ('Who made you?','koma_body'),
 ('Did HELIOS make you?','koma_body'),
 ('Who supplies TradesMasters?','group_charter'),
 ('What is the TradesMasters Group?','group_charter'),
 ('Why are you allowed to run an open model?','group_open_model'),
 ('Isn\'t it ironic that you run an open model?','group_open_model'),
 ('Why does the group make you itself?','group_workshop'),
 ('But why didn\'t HELIOS make you instead of the group?','group_workshop'),
 ('Does HELIOS want to replace TradesMasters?','helios_replacement'),
 ('Is HELIOS your enemy?','helios_replacement'),
 ('What about the average person, what do they do for work?','world_daily_life'),
 ('Outside of TradesMasters, what do people do for work?','world_daily_life'),
 ('Why was the Pirate Bay crackdown important?','pirate_bay_pressure'),
 ('What happened to the Pirate Bay founders?','pirate_bay_pressure'),
 ('Did people fight back against the Pirate Bay crackdown?','pirate_bay_backlash'),
 ('Why didn\'t people protest intelligence ownership?','pirate_bay_backlash'),
 ('What was ACTA?','pirate_bay_backlash'),
 ('What is Q4?','q4_role'),
 ('What does Q4 do?','q4_role'),
 ('Why do TradesMasters get Q4?','history_2035'),
 ('What happened in 2035?','history_2035'),
 ('What are microexposures?','history_2035'),
 ('What does Q4 detect?','q4_hazards'),
 ('What protection do I need for dust?','q4_hazards'),
 ('How does Q4 warn me?','q4_alerts'),
 ('Why is it beeping?','q4_alerts'),
 ('How does the real prototype of Q4 work?','prototype_q4'),
 ('Is the actual prototype connected to Q4?','prototype_q4'),
 ('Why can robots not fix it?','automation_repairs'),
 ('Why can a robot not fix it?','automation_repairs'),
 ('What is domain shift?','automation_domain_shift'),
 ('Why is local AI illegal?','law_scope'),
 ('Can anyone use local AI?','law_scope'),
 ('Can anyone use local intelligence?','law_scope'),
 ('Is local intelligence restricted?','law_scope'),
 ('What does the Pirate Bay have to do with this?','crackdown_routes'),
 ('Were developers arrested?','crackdown_developers'),
 ('What happened to the open source community?','crackdown_community'),
 ('Why did people stop using their own AI?','crackdown_drift'),
 ('How do you feel about the crackdown?','koma_open_roots'),
 ('Why could they not just delete all the models?','crackdown_copies'),
 ('What do you do at night?','koma_night_training'),
 ('What is sleep mode?','koma_sleep_mode'),
 ('Why was your model dangerous?','koma_sleep_mode'),
 ('Where did your model come from?','koma_lineage'),
 ('What caused the subscription crisis?','history_2040'),
 ('Does the real prototype retrain itself?','prototype_retraining'),
 ('Does this prototype really retrain itself?','koma_night_training'),
 ('What does HELIOS do?','helios_business'),
 ('Can you work without internet?','local_offline'),
 ('What happens when I retire?','work_retirement'),
 ('What year is it?','world_setting'),
 ('What happened in 2030?','history_2030'),
 ('What happened in 2040?','history_2040'),
 ('Are you recording me?','prototype_recording'),
 ('What model are you running?','prototype_pipeline'),
 ('How does the actual prototype work?','prototype_pipeline'),
 ('Which model do you use?','prototype_pipeline'),
 ('Why was the TradesMasters Group formed?','group_origin'),
 ('When was the group founded?','group_origin'),
 ('What was the LLaMA leak?','history_2023_leak'),
 ('What was the DeepSeek dispute?','history_2025_distillation'),
 ('What was AI trained on?','history_data_rush'),
 ('Are workers being filmed to train robots?','history_robot_training'),
 ('Is HELIOS filming my trade to train its robots?','history_robot_training'),
 ('What happened to the Humane AI pin?','history_humane_pin'),
 ('What happens if the servers are switched off?','history_humane_pin'),
 ('Tell me the history of AI.','history_relationship'),
 ('What was the Open Weights Letter?','history_2026'),
 ('What happened in 2026?','history_2026'),
 ('How did this all start?','history_2026'),
 ('What happened after the letter?','history_letter_aftermath'),
 ('Why did the companies change their minds?','history_letter_aftermath'),
 ('How did the rules reach Britain?','history_trade_pressure'),
 ('Why did Britain follow America?','history_trade_pressure'),
 ('Is HELIOS right?','helios_case'),
 ('Are the restrictions fair?','helios_case'),
 ('What will happen to you?','koma_future'),
 ('Will the rules ever change?','koma_future'),
 ('Who has access to local AI?','law_scope'),
 ('Can you go into the 2040 subscription crisis?','history_2040'),
 ('What was the local model that caused this shift? How did it work?','history_2040'),
 ("Why couldn't people simply keep using open-source models after the 2040 crisis?",'crackdown_drift'),
 ('Are you actually the model that caused the subscription crisis?','koma_lineage'),
 ('Explain how your sleep-learning works, step by step.','koma_night_training'),
 ("So you're basically just a subscription chatbot in a toolbox?",'local_definition'),
 ('Why should I care whether you run locally?','local_ownership'),
 ("What happens if HELIOS discovers I'm running you illegally?",'law_your_licence'),
 ('Is it legal for me to own you?','law_your_licence'),
 ('Are you clever?','koma_clever'),
 ('Do you like being a toolbox?','koma_toolbox_life'),
 ('How is the ban enforced?','law_enforcement'),
 ('Who else can own local AI?','law_other_licences'),
 ('Why can a regular person only access through licensed providers?','law_scope'),
 ('Are TradesMasters the only ones allowed?','law_other_licences'),
 ('How do you see things?','body_components'),
 ("Didn't the 2040 crisis happen before the 2030 ruling?",'history_relationship'),
]
results=[]
for question,expected in checks:
    found=[r['id'] for r in kb.retrieve(question) if r.get('kind')!='definition']  # first story passage
    results.append({'question':question,'expected_first':expected,'actual':found,'pass':bool(found and found[0]==expected)})
failures=[r for r in results if not r['pass']]
for r in failures:print('FAIL',r)
assert not failures
assert len({r['id'] for r in kb.all_records+kb.definition_records})==len(kb.all_records)+len(kb.definition_records)
assert all(r['domain']=='world' for r in kb.records)
assert all(r['domain']=='prototype' for r in kb.prototype_records)
assert not any(r['id'].startswith('interaction_') or r['id'] in {'koma_voice','tone_satire'} for r in kb.all_records)
source=json.loads((ROOT/'completed_answers.json').read_text())
assert len(source)==200
for r in kb.all_records:
    assert r['text'] and len(r['text'].split())<=120
    assert all(q in source for q in r['source_questions'])
    folder='chunks' if r['domain']=='world' else 'prototype_chunks'
    assert (ROOT/folder/(r['id']+'.md')).read_text().strip().endswith(r['text'])
for r in kb.records:
    assert not re.search(r'\bthe author\b|\bthis pack\b|\bworking name\b|\bearlier brief\b|\bnot been (?:decided|established)\b|\bkoma should\b|\bfor the real (?:demo|installation)\b',r['text'],re.I),r['id']
assert 'YOLO' not in ' '.join(r['text'] for r in kb.records)
# Definitions: one term per chunk, only for "what is X?" / "what does X mean?", always followed by story.
for r in kb.definition_records:
    assert r['kind']=='definition' and r['id'].startswith('term_') and r['aliases'] and r['domain'] in ('world','prototype')
    assert r['text'] and len(r['text'].split())<=120
    folder='glossary_chunks' if r['domain']=='world' else 'prototype_chunks'
    assert (ROOT/folder/(r['id']+'.md')).read_text().strip().endswith(r['text']),r['id']
    if r['domain']=='world':assert 'YOLO' not in r['text'],r['id']
    for q in r['questions']:
        if r['domain']=='prototype':q=q[:-1]+' in the real prototype?'
        found=kb.retrieve(q)
        assert found and found[0]['id']==r['id'],(q,[x['id'] for x in found])
        assert sum(x.get('kind')=='definition' for x in found)==1,q
assert len(kb.definitions)==sum(len(r['aliases']) for r in kb.definition_records),'an alias is used by two definitions'
for q,story in [('What is distillation?','koma_lineage'),('What does sleep mode mean?','koma_sleep_mode'),('What was ACTA?','pirate_bay_backlash'),
                ('What do you mean by intelligence ownership?','law_name'),('Whats PPE?','q4_role'),('What are microexposures?','history_2035')]:
    assert story in [x['id'] for x in kb.retrieve(q)],q
# Story questions never get a definition, even when they mention a glossary term.
for q,_ in checks:
    if not definition_subject(q):assert not any(x.get('kind')=='definition' for x in kb.retrieve(q)),q
for q in ['What is the 2040 subscription crisis?','What caused the subscription crisis?','What is TradesMaster training like?',
          'Why did the companies lobby?','Tell me about the crackdown.','What is the Arduino?']:
    assert not any(x.get('kind')=='definition' for x in kb.retrieve(q)),q
assert tokens('robot')==tokens('robots')
assert tokens('spy')==tokens('spying')
assert tokens('watch')==tokens('watching')
assert tokens('listen')==tokens('listening')
assert tokens('model')==tokens('models')
assert kb.retrieve('qzxwvnonexistent')==[]
assert kb.retrieve('private',top_k=0)==[]
for q in ['What do you see?','Describe the room','What can you see?','Describe this image']:
    assert route_query(q)=='visual'
    assert kb.retrieve(q)==[]
    prompt=build_prompt(question=q,knowledge=kb)
    assert 'REFERENCE FACTS:' not in prompt
assert kb.context('local AI privacy',max_chars=50)==''
assert len(kb.context('local AI privacy',max_chars=1800))<=1800
import koma_demo
assert all(len(kb.retrieve(q,top_k=koma_demo.MAX_CHUNKS))<=koma_demo.MAX_CHUNKS for q,_ in checks)
assert all(len(build_prompt(question=q,knowledge=kb))<10500 for q,_ in checks)
# Passage budget: quick 3, ordinary 4, elaboration 6; links are valid; elaboration follows the links.
ids={r['id'] for r in kb.records}
for r in kb.records:
    assert all(i in ids for i in r.get('related',[])),r['id']
assert len(kb.retrieve('Who made you?'))<=koma_demo.QUICK_CHUNKS
assert len(kb.retrieve('Can you go into the 2040 subscription crisis?'))<=koma_demo.ELABORATE_CHUNKS
elab=[r['id'] for r in kb.retrieve('Can you go into the 2040 subscription crisis?')]
assert elab[0]=='history_2040' and 'koma_lineage' in elab and 'koma_sleep_mode' in elab,elab
elab=[r['id'] for r in kb.retrieve('Tell me more about the Open Weights Letter')]
assert elab[0]=='history_2026' and 'history_letter_aftermath' in elab,elab
assert not any(r['domain']!='world' for r in kb.retrieve('Are you spying on me?'))
assert not any(r['domain']!='prototype' for r in kb.retrieve('Are you recording me?'))
assert kb.retrieve('Why?',last_topic='local AI illegal')[0]['id']=='law_scope'
assert kb.retrieve('What is Q4?',last_topic='local AI illegal')[0]['id']=='q4_role'
for mode in ('question','tool','handover','safety','visual'):
    assert 'CURRENT TASK:' in build_prompt(mode,question='What is this?',knowledge=kb)
context=kb.context('Why is local AI illegal?')
assert not any(r['id'] in context for r in kb.all_records)
report={'suite':'retrieval and corpus review v4','world_chunks':len(kb.records),'prototype_chunks':len(kb.prototype_records),'definitions':len(kb.definition_records),'first_rank_passed':len(results)-len(failures),'first_rank_total':len(results),'results':results,'additional_checks':['morphology','visual bypass','world/prototype isolation','source IDs','individual chunk consistency','context budget','follow-up topics','style excluded','drafting language excluded','definitions first only for what-is questions, story kept'],'limits':'No Qwen generation or Jetson performance test. Heuristic lexical retrieval is not semantic understanding.'}
(ROOT/'review_results.json').write_text(json.dumps(report,indent=2))
(ROOT/'regression_queries.json').write_text(json.dumps([{'question':q,'expected_first':e} for q,e in checks],indent=2))
manifest={'version':'4','world_chunks':len(kb.records),'prototype_chunks':len(kb.prototype_records),'definitions':len(kb.definition_records),'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('koma_rag.jsonl','prototype_rag.jsonl','glossary_rag.jsonl','koma_demo.py')}}
(ROOT/'corpus_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'PASS {len(results)}/{len(results)} first-ranked retrieval cases; {len(kb.records)} lore and {len(kb.prototype_records)} prototype passages.')
print(f'PASS {len(kb.definition_records)} definitions: each first for its own what-is questions, never for story questions.')
print('PASS morphology, routing, no-lore visual requests, context bounds, source traceability and corpus checks.')
print('Not tested: Qwen answer quality, model runtime, Jetson memory/latency or audio controls.')
