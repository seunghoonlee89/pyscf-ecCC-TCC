#!/bin/bash

awk '{ 
  if($1=="CI" && $2=="coeff" && $3=="start")
    {
       i=0;
       while ( 1 == 1 )
       {
        getline;
        if ($0==" CI coeff end "){break;}
        i = i+1;
        line[i] = $0;   
       }
    }
  else if($1=="Final" && $2=="iteration")
    {
      energy_shci = $9; 
    }

 } END { 
    for (j=1;j<=i;j++){
        printf("%s\n",line[j]);
    }
    printf("shci,%15.10f\n",energy_shci);
 }' "$1" > CIcoeff_shci.out 

