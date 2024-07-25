#!/bin/bash
MESG="commit"
git pull ; git add . --all; git commit -m "$MESG" ; git push origin main