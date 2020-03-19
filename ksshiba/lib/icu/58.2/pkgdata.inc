GENCCODE_ASSEMBLY_TYPE=-a gcc-darwin
SO=dylib
SOBJ=dylib
A=a
LIBPREFIX=lib
LIB_EXT_ORDER=.58.2.dylib
COMPILE=/Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/bin/clang -DU_ATTRIBUTE_DEPRECATED= -D_FORTIFY_SOURCE=2 -mmacosx-version-min=10.9   -DU_HAVE_ATOMIC=1 -DU_HAVE_STRTOD_L=1  -march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -std=c99 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -fno-common -c
LIBFLAGS=-I/Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/include  -dynamic
GENLIB=/Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/bin/clang -dynamiclib -dynamic -march=core2 -mtune=haswell -mssse3 -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -std=c99 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -pie -headerpad_max_install_names -Wl,-rpath,/Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/lib 
LDICUDTFLAGS=
LD_SONAME=-Wl,-compatibility_version -Wl,58 -Wl,-current_version -Wl,58.2 -install_name /Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/lib/
RPATH_FLAGS=
BIR_LDFLAGS=
AR=/Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/bin/x86_64-apple-darwin13.4.0-ar
ARFLAGS=r -c
RANLIB=/Users/alexjorguer/Documents/GITHUB/K-SSHIBA/ksshiba/bin/x86_64-apple-darwin13.4.0-ranlib
INSTALL_CMD=/usr/bin/install -c
