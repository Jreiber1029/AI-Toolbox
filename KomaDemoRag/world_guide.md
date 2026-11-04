"""Retrieval/routing regressions; does not test Qwen generation or hardware."""
import hashlib
import json
from pathlib import Path
import re
from koma_demo import KomaKnowledge, build_prompt, tokens, route_query

ROOT=Path(__file__).resolve().parent
kb=KomaKnowledge()
checks=[
 ('Are you spying on me?','local_privacy'),
 ('What is this thing on my wrist?','ppes_role'),
 ('How long do I have to work?','work_conditions'),
 ('What tools do I need for plumbing?','work_examples'),
 ('Why are you different from ChatGPT?','local_definition'),
 ('So what are you exactly?','koma_body'),
 ('Are you keeping an eye on me?','local_privacy'),
 ('Is my conversation private?','local_privacy'),
 ('Will someone else hear what I say?','local_privacy'),
 ('What does the wrist device do?','ppes_role'),
 ('Explain this wearable to me.','ppes_role'),
 ('How many hours is a shift?','work_conditions'),
 ('Can I choose my work schedule?','work_conditions'),
 ('Which plumbing tools should I carry?','work_examples'),
 ('What would I use to repair pipes?','work_examples'),
 ('How do you compare to a cloud chatbot?','local_definition'),
 ('What makes you unlike ChatGPT?','local_definition'),
 ('Tell me about yourself.','koma_body'),
 ('Who made you?','koma_body'),
 ('Why can robots not fix it?','automation_repairs'),
 ('Why can a robot not fix it?','automation_repairs'),
 ('What is domain shift?','automation_domain_shift'),
 ('Why is local AI illegal?','law_scope'),
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
]
results=[]
for question,expected in checks:
    found=[r['id'] for r in kb.retrieve(question)]
    results.append({'question':question,'expected_first':expected,'actual':found,'pass':bool(found and found[0]==expected)})
failures=[r for r in results if not r['pass']]
for r in failures:print('FAIL',r)
assert not failures
assert len({r['id'] for r in kb.all_records})==len(kb.all_records)
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
assert not any(r['domain']!='world' for r in kb.retrieve('Are you spying on me?'))
assert not any(r['domain']!='prototype' for r in kb.retrieve('Are you recording me?'))
assert kb.retrieve('Why?',last_topic='local AI illegal')[0]['id']=='law_scope'
assert kb.retrieve('What is PPES?',last_topic='local AI illegal')[0]['id']=='ppes_role'
for mode in ('question','tool','handover','safety','visual'):
    assert 'CURRENT TASK:' in build_prompt(mode,question='What is this?',knowledge=kb)
context=kb.context('Why is local AI illegal?')
assert not any(r['id'] in context for r in kb.all_records)
report={'suite':'retrieval and corpus review v4','world_chunks':len(kb.records),'prototype_chunks':len(kb.prototype_records),'first_rank_passed':len(results)-len(failures),'first_rank_total':len(results),'results':results,'additional_checks':['morphology','visual bypass','world/prototype isolation','source IDs','individual chunk consistency','context budget','follow-up topics','style excluded','drafting language excluded'],'limits':'No Qwen generation or Jetson performance test. Heuristic lexical retrieval is not semantic understanding.'}
(ROOT/'review_results.json').write_text(json.dumps(report,indent=2))
(ROOT/'regression_queries.json').write_text(json.dumps([{'question':q,'expected_first':e} for q,e in checks],indent=2))
manifest={'version':'4','world_chunks':len(kb.records),'prototype_chunks':len(kb.prototype_records),'files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('koma_rag.jsonl','prototype_rag.jsonl','koma_demo.py')}}
(ROOT/'corpus_manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'PASS {len(results)}/{len(results)} first-ranked retrieval cases; {len(kb.records)} lore and {len(kb.prototype_records)} prototype passages.')
print('PASS morphology, routing, no-lore visual requests, context bounds, source traceability and corpus checks.')
print('Not tested: Qwen answer quality, model runtime, Jetson memory/latency or audio controls.')
