#!/bin/bash

mydir=$(readlink -f $( dirname $0 ) )

cat <<END_JSON
{
   "fake_addon" : "export PATH=${mydir}/bin:\${PATH}"
}
END_JSON
