#!/bin/bash

fqdn=$(hostname -f)

for prefix in $(cat ./hdfs-prefix)
do
    exist=`cat ./princ | grep $prefix/$fqdn`
    if [ -z "$exist" ]
        then
            echo "addprinc -randkey $prefix/$fqdn" | cat >> ./princ
            echo "ktadd -k /etc/security/keytab/$prefix.service.keytab $prefix/$fqdn" | cat >> ./princ
    fi
done


exist=`cat ./princ | grep HTTP/$fqdn`
if [ -z "$exist" ]
then
    echo "addprinc -randkey HTTP/$fqdn" | cat >> ./princ
    echo "ktadd -k /etc/security/keytab/spnego.service.keytab HTTP/$fqdn" | cat >> ./princ
fi

fqdn=resourcemanager.ustc.edu

for prefix in $(cat ./yarn-prefix)
do
    exist=`cat ./princ | grep $prefix/$fqdn`
    if [ -z "$exist" ]
        then
            echo "addprinc -randkey $prefix/$fqdn" | cat >> ./princ
            echo "ktadd -k /etc/security/keytab/$prefix.service.keytab $prefix/$fqdn" | cat >> ./princ
    fi
done

exist=`cat ./princ | grep HTTP/$fqdn`
if [ -z "$exist" ]
then
    echo "addprinc -randkey HTTP/$fqdn" | cat >> ./princ
    echo "ktadd -k /etc/security/keytab/spnego.service.keytab HTTP/$fqdn" | cat >> ./princ
fi
#cat ./princ | kadmin.local

