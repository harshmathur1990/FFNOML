#include <stdio.h>
#include <stdlib.h>
double cubeconvol(int Nx, int Ny, double *f, double x, double y);
int main(int argc,char**argv){
  /* read N1 N2 then N1*N2 values (row-major, rows = neff1), then pairs x y; print values */
  int n1,n2; scanf("%d %d",&n1,&n2); double *f=malloc(sizeof(double)*n1*n2);
  for(int i=0;i<n1*n2;i++) scanf("%lf",&f[i]);
  double x,y; while(scanf("%lf %lf",&x,&y)==2) printf("%.15e\n",cubeconvol(n2,n1,f,x,y));
  return 0;}
