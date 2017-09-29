#!/bin/bash
./ac 2000 $((RANDOM%100)) > acyclic-dense-2000
./ac 4000 $((RANDOM%100)) > acyclic-dense-4000
./ac 6000 $((RANDOM%100)) > acyclic-dense-6000
