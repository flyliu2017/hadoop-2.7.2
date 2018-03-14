#!/usr/bin/env bash

if [ $# -lt 2 ]
then
    echo "need two arg: principal_name  keytab_name"
else
    echo -e "addprinc -randkey $1 \n ktadd -k /etc/security/keytab/$2 $1" | kadmin.local
fi

