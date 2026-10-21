#!/usr/bin/env python3
"""Single fixed-size calibration extension. Preserve failed pilot verbatim."""
import hashlib
import json
import shutil
import run_campaign as c

root=c.ROOT
record=c.RESULTS/'execution.json'
r=json.loads(record.read_text())
if r['completed'] or 'calibration_extension_started' in r:
    raise RuntimeError('extension requires original incomplete pilot; never silently rerun')
for name in ['calibration.csv','execution.json']:
    dst=c.RESULTS/('pilot-'+name)
    if dst.exists(): raise RuntimeError('pilot copy exists')
    shutil.copy2(c.RESULTS/name,dst)
addition=json.loads((root/'calibration_addendum.json').read_text())
r['calibration_extension_started']=True
r['calibration_addendum_sha256']=hashlib.sha256((root/'calibration_addendum.json').read_bytes()).hexdigest()
r['calibration_extension_script_sha256']=hashlib.sha256((root/'calibrate_extension.py').read_bytes()).hexdigest()
record.write_text(json.dumps(r,indent=2)+'\n')
c.P['calibration']['trials_per_source']=addition['trials_per_source']
original_seed=c.seed
c.seed=lambda namespace,group,replicate=0:original_seed(namespace+addition['namespace_suffix'],group,replicate)
c.calibration()
r['completed']=['calibration']
r['calibration_basis']='independent 4096-trial extension; initial failed gate retained in pilot-calibration.csv'
record.write_text(json.dumps(r,indent=2)+'\n')
print('Independent calibration extension passed',flush=True)
