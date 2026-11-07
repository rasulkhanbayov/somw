/* roundeven/roundevenf/roundevenl are glibc >= 2.25; the cluster's cpu-all nodes and the 2.17 sysroot lack them.
   rint rounds half to even in the default rounding mode, which is the mode PolyFEM runs in. */
#include <math.h>
double roundeven(double x) { return rint(x); }
float roundevenf(float x) { return rintf(x); }
long double roundevenl(long double x) { return rintl(x); }
