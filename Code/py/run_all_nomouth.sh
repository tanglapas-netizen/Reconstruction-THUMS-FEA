#!/bin/bash
# re-run every usable case (except case_09) skin-only without the mouth interior, 4 at a time
cd "$(dirname "$0")"; export OUT_TAG=_nomouth
ls ../../Data/outputs/hanseg_template_reg_skinonly | grep '^case_' | xargs -P 4 -I{} sh -c 'python3 hanseg_template_registration.py {} --skin-only > /tmp/nm/{}.out 2>&1'
echo done > /tmp/nm/ALL_DONE
