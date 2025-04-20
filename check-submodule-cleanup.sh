#!/bin/bash

echo "✅ Check if .gitmodules exists"
[ -f .gitmodules ] && echo "❌ .gitmodules exists" || echo "✅ No .gitmodules"

echo "✅ Check if git tracks .gitmodules"
git ls-files .gitmodules >/dev/null 2>&1 && echo "❌ Still tracked" || echo "✅ Not tracked"

echo "✅ Check submodule config in git"
git config --list | grep submodule && echo "❌ Submodule config exists" || echo "✅ No submodule config"

echo "✅ Check if yolov5 is a plain directory"
[ -d yolov5/.git ] && echo "❌ yolov5 is still a git repo" || echo "✅ yolov5 is plain directory"

echo "✅ Check .git/modules"
[ -d .git/modules/yolov5 ] && echo "❌ .git/modules/yolov5 still exists" || echo "✅ No leftover in .git/modules"

