#!/bin/bash
MESG="commit"
git pull ; git add . ; git commit -m "$MESG" ; git push origin main