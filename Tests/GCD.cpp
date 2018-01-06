#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <typeinfo>

#include "QPULib.h"

void gcd(Ptr<Int> p, Ptr<Int> q, Ptr<Int> r)
{
  Int a = *p;
  Int b = *q;
  While (any(a != b))
  Where (a > b)
    a = a-b;
  End
  Where (a < b)
    b = b-a;
  End
  End
  *r = a;
}

void gpu_dot_improve(Ptr<Float> p, Ptr<Float> q, Ptr<Float> r, Int A, Int B, Int C)
{
  const int BUFFER = 3;
  Float a, b, acc[16];
  Ptr<Float> pstart = p + index();
  Ptr<Float> qstart = q + index();
  Ptr<Float> rstart = r + index();
  For(Int q_cols = me()*16, q_cols < C, q_cols=q_cols+16*numQPUs())
    For(Int p_cols=0, p_cols < A, p_cols=p_cols+16)
      r = rstart + p_cols*C + q_cols;
      p = pstart + p_cols;
      q = qstart + q_cols;
      for(int z = 0; z < 16; z++){
        acc[z] = 0;
      }
      Int rows;
      For(rows=0, rows+BUFFER < B, rows=rows+BUFFER)
        for(int z = 0; z < BUFFER; z++){
          gather(p+A*z); gather(q+C*z);
        }
        for(int z = 0; z < BUFFER; z++){
          receive(a); receive(b);
          for(int k = 0; k < 16; k++){
            acc[k] = acc[k] + a*rotate(b,(16-k)%16);
          }
        }
        p = p + A*BUFFER; q = q + C*BUFFER;
      End
      For(, rows < B, rows=rows+1)
        gather(p); gather(q);
        receive(a); receive(b);
        for(int k = 0; k < 16; k++){
          acc[k] = acc[k] + a*rotate(b,(16-k)%16);
        }
        p = p + A; q = q + C;
      End
      for(int i = 0; i < 16; i++){
        for(int j = 0; j < 16; j++){
          Where (index() == j)
            a = rotate(acc[(16-i+j)%16], (16-i+j)%16);
          End
        }
        store(a, r);
        r = r+C;
      }
    End
  End
}

void swap(float& a, float& b) {
  float temp = a;
  a = b;
  b = temp;
}

float dot(float* a, float* b, int n) {
  float sum = 0;
  for( int i = 0; i < n; i++ ){
    sum += a[i]*b[i];
  }
  return sum;
}

void matrix_mult(float* a, float* b, float* r, int m, int n) {
  for( unsigned int i = 0; i < m; i++ ) {
    for( unsigned int j = 0; j < n; j++ ) {
      r[i*n+j] = dot(a+i*n, b+j*n, n);
    }
  }
}

void transpose(float* a, int N){
  for( int i = 0; i < N; i++ ){
    for( int j = 1; j < N-i; j++ ){
      swap(a[i*N+i+j], a[i*N+i+j*N]);
    }
  }
}

void matrix_mult_gpu(float* a, float* b, float* c, int m, int n) {
  SharedArray<float> p(m*n);
  SharedArray<float> q(m*n);
  SharedArray<float> r(n*n);
  auto k = compile(gpu_dot_improve);
  k.setNumQPUs(12);
  //transpose(b, n);
  for( int i = 0; i < m*n; i++ ){
    p[i] = a[i];
    q[i] = b[i];
  }
  auto wcts = std::chrono::system_clock::now();
  for(int i = 0; i < 1; i++)
    k(&p, &q, &r, m, n, n);
  std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
  std::cout << "GPU Finished in " << wctduration.count() << " seconds [Wall Clock]" << std::endl;
  std::cout << "Type is: " << typeid(k).name() << std::endl;
  for( int i = 0; i < n*n; i++ ){
    c[i] = r[i];
  }
  //fix(c, m, n);
}

void test_cpu() {
  srand (0);
  /* generate secret number between 1 and 10: */
  //iSecret = rand() % 10 + 1;
  const int M = 512;
  const int N = 512;
  float a[M*N], b[M*N], r[N*N], gr[N*N+16];
  for( int j = 0; j < M; j++ ){
    for( int i = 0; i < N; i++ ){
      a[j*N+i] = b[j*N+i] = static_cast<float>(rand()%10);
    }
  }
  printf("START\n");
  transpose(b, N);
  /*{
    auto wcts = std::chrono::system_clock::now();
    matrix_mult(a, b, r, M, N);
    std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout << "CPU Finished in " << wctduration.count() << " seconds [Wall Clock]" << std::endl;
  }*/
  transpose(a, N);
  transpose(b, N);
  {
    matrix_mult_gpu(a, b, gr, M, N);
  }
  bool flag = false;
  int counter = 0;
  //printf("CPU\n\n");
  for( int i = 0; i < N; i++ ){
    for( int j = 0; j < N; j++ ){
      if( gr[i*N+j] != r[i*N+j] ){
        flag = true;
        counter++;
      }
    }
  }
  if( flag == true ){
    printf("Not Correct.\n");
  }
  printf("%f\n", r[N*N-1]);
  printf("%f\n", gr[N*N-1]);
  printf("Differences: %d\n", counter);
}

int main()
{
  test_cpu();
  return 0;
}
