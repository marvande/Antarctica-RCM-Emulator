
nb="S1"

for var in  SMB SF ; do #SF ME SMB RF SU RU ['SWD','LWD','SWN','AL2','SHF','LHF']  RU SMB SF ME SU RF
rm -f fig${nb}_${var}.png     
figa="fig${nb}_${var}_AC3.png"
figb="fig${nb}_${var}_NOR.png"

figc="fig${nb}_${var}_CN6.png"
figd="fig${nb}_${var}_CSM.png"
cb="cb_${var}_pre.png"

convert $figa $figb +append fig${nb}_tmp1.png
convert $figc $figd +append fig${nb}_tmp2.png
convert fig${nb}_tmp1.png fig${nb}_tmp2.png -append fig${nb}_${var}.png

convert fig${nb}_${var}.png $cb +append fig${nb}_${var}.png2
mv -f fig${nb}_${var}.png2 fig${nb}_${var}.png
rm -f  fig${nb}_tmp1.png fig${nb}_tmp2.png
done


#for var in TT ; do #['SWD','LWD','SWN','AL2','SHF','LHF']  RU SMB SF ME SU RF
#rm -f fig${nb}_${var}_DJF.png     
#figa="fig${nb}_${var}_AC3_DJF.png"
#figb="fig${nb}_${var}_NOR_DJF.png"
#
#figc="fig${nb}_${var}_CN6_DJF.png"
#figd="fig${nb}_${var}_CSM_DJF.png"
#cb="cb_${var}_pre.png"
#
#convert $figa $figb +append fig${nb}_tmp1.png
#convert $figc $figd +append fig${nb}_tmp2.png
#convert fig${nb}_tmp1.png fig${nb}_tmp2.png -append fig${nb}_${var}_DJF.png
#
#convert fig${nb}_${var}_DJF.png $cb +append fig${nb}_${var}.png2
#mv -f fig${nb}_${var}.png2 fig${nb}_${var}_DJF.png
#rm -f  fig${nb}_tmp1.png fig${nb}_tmp2.png
#done
