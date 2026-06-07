#!/bin/bash
# cd /home/sehwan001/projects/deepthaw && bash setup_all.sh
set -e
echo "=== DeepThaw Setup ==="

# 1. select_masking 패치
echo ">>> Patching select_masking..."
python3 -c "
f='uvcgan2/torch/image_masking.py'
code=open(f).read()
if 'def select_masking' not in code:
    open(f,'a').write('''

def select_masking(masking):
    if masking is None:
        return None
    raise ValueError(\"Unknown masking: %s\" % masking)
''')
    print('  PATCHED')
else:
    print('  Already exists')
"

# 2. 파일 배치
echo ">>> Deploying files..."
[ -f uvcgan2_deepthaw.py ] && cp uvcgan2_deepthaw.py uvcgan2/cgan/ && echo "  ✅ uvcgan2_deepthaw.py"
[ -f cgan__init__.py ] && cp cgan__init__.py uvcgan2/cgan/__init__.py && echo "  ✅ cgan__init__.py"

# 3. 검증
echo ">>> Verify..."
python3 -c "from uvcgan2.torch.image_masking import select_masking; print('  ✅ select_masking OK')"
python3 -c "from uvcgan2.cgan import CGAN_MODELS; print('  ✅ Models:', sorted(CGAN_MODELS.keys()))"

echo ""
echo "=== Done! ==="
