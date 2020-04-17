
<?php

require_once('library/Apriori/lib/Apriori.class.php');

//variables
$minSupp  = 5;                  //minimal support
$minConf  = 75;                 //minimal confidence
$type     = Apriori::SRC_PLAIN; //data type