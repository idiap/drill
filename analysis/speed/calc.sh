#    Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of drill which builds over awd-lstm-lm codebase
#    (https://github.com/salesforce/awd-lstm-lm).
#
#    drill is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    drill is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with drill. If not, see http://www.gnu.org/licenses/

echo "Penn\n===="
export base=`cat penn/penn-base.txt | awk {'print $8'} | awk {'split($0, t, "s"); print t[1]'} | awk {' total += $0; count++ } END {print total/count}'`
base=`printf '%.2f' "$base"`
echo "base", $base, 1.00 'x'

export ours=`cat penn/penn-ours.txt | awk {'print $8'} | awk {'split($0, t, "s"); print t[1]'} | awk {' total += $0; count++ } END {print total/count}'`
ours=`printf '%.2f' "$ours"`
export dif=`echo "scale=3; $ours / $base" | bc -l`
echo "ours", $ours, $dif 'x'

export mos=`cat penn/penn-mos.txt | awk {'print $8'} | awk {'split($0, t, "s"); print t[1]'} | awk {' total += $0; count++ } END {print total/count}'`
mos=`printf '%.2f' "$mos"`
export dif=`echo "scale=3; $mos / $base" | bc -l`
echo "mos", $mos, $dif 'x'

echo "Wiki2\n===="
export wbase=`cat wiki2/wiki2-base.txt | awk {'print $8'} | awk {'split($0, t, "s"); print t[1]'} | awk {' total += $0; count++ } END {print total/count}'`
wbase=`printf '%.2f' "$wbase"`
echo "base", $wbase, 1.00 'x'

export wours=`cat wiki2/wiki2-ours.txt | awk {'print $8'} | awk {'split($0, t, "s"); print t[1]'} | awk {' total += $0; count++ } END {print total/count}'`
wours=`printf '%.2f' "$wours"`
export wdif=`echo "scale=3; $wours / $wbase" | bc -l`
echo "ours", $wours, $wdif 'x'

export wmos=`cat wiki2/wiki2-mos.txt | awk {'print $8'} | awk {'split($0, t, "s"); print t[1]'} | awk {' total += $0; count++ } END {print total/count}'`
wmos=`printf '%.2f' "$wmos"`
export wdif=`echo "scale=3; $wmos / $wbase" | bc -l`
echo "mos", $wmos, $wdif 'x'
