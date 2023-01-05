! This automatically generated Fortran wrapper file allows codes
! written in Fortran to be called directly from C and translates all
! C-style arguments into expected Fortran-style arguments (with
! assumed size, local type declarations, etc.).


MODULE C_VTMOP_LIBE_MOD
  IMPLICIT NONE


CONTAINS



  SUBROUTINE C_VTMOP_LIBE_INIT(D, P, LB_DIM_1, LB, UB_DIM_1, UB, IERR, LOPT_BUDGET, DECAY, DES_TOL, EPS, EPSW, OBJ_TOL, MIN_RADF, T&
&RUST_RADF, OBJ_BOUNDS_DIM_1, OBJ_BOUNDS_DIM_2, OBJ_BOUNDS, PMODE) BIND(C)
    USE VTMOP_MOD
    USE ISO_FORTRAN_ENV
    USE VTMOP_LIBE_MOD, ONLY: VTMOP_LIBE_INIT
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: D
    INTEGER, INTENT(IN) :: P
    INTEGER, INTENT(IN) :: LB_DIM_1
    REAL(KIND=R8), INTENT(IN), DIMENSION(LB_DIM_1) :: LB
    INTEGER, INTENT(IN) :: UB_DIM_1
    REAL(KIND=R8), INTENT(IN), DIMENSION(UB_DIM_1) :: UB
    INTEGER, INTENT(OUT) :: IERR
    INTEGER, INTENT(IN) :: LOPT_BUDGET
    REAL(KIND=R8), INTENT(IN) :: DECAY
    REAL(KIND=R8), INTENT(IN) :: DES_TOL
    REAL(KIND=R8), INTENT(IN) :: EPS
    REAL(KIND=R8), INTENT(IN) :: EPSW
    REAL(KIND=R8), INTENT(IN) :: OBJ_TOL
    REAL(KIND=R8), INTENT(IN) :: MIN_RADF
    REAL(KIND=R8), INTENT(IN) :: TRUST_RADF
    INTEGER, INTENT(IN) :: OBJ_BOUNDS_DIM_1
    INTEGER, INTENT(IN) :: OBJ_BOUNDS_DIM_2
    REAL(KIND=R8), INTENT(IN), DIMENSION(OBJ_BOUNDS_DIM_1,OBJ_BOUNDS_DIM_2) :: OBJ_BOUNDS
    LOGICAL, INTENT(IN) :: PMODE

    CALL VTMOP_LIBE_INIT(D, P, LB, UB, IERR, LOPT_BUDGET, DECAY, DES_TOL, EPS, EPSW, OBJ_TOL, MIN_RADF, TRUST_RADF, OBJ_BOUNDS, PMO&
&DE)
  END SUBROUTINE C_VTMOP_LIBE_INIT



  SUBROUTINE C_VTMOP_LIBE_GENERATE(D, P, LB_DIM_1, LB, UB_DIM_1, UB, DES_PTS_DIM_1, DES_PTS_DIM_2, DES_PTS, OBJ_PTS_DIM_1, OBJ_PTS_&
&DIM_2, OBJ_PTS, ISNB, SNB, ONB, OBJ_BOUNDS_DIM_1, OBJ_BOUNDS_DIM_2, OBJ_BOUNDS, LBATCH, BATCHX_DIM_1, BATCHX_DIM_2, BATCHX, IERR) &
&BIND(C)
    USE VTMOP_MOD
    USE ISO_FORTRAN_ENV
    USE ISO_FORTRAN_ENV, ONLY: INT64
    USE VTMOP_LIBE_MOD, ONLY: VTMOP_LIBE_GENERATE
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: D
    INTEGER, INTENT(IN) :: P
    INTEGER, INTENT(IN) :: LB_DIM_1
    REAL(KIND=R8), INTENT(IN), DIMENSION(LB_DIM_1) :: LB
    INTEGER, INTENT(IN) :: UB_DIM_1
    REAL(KIND=R8), INTENT(IN), DIMENSION(UB_DIM_1) :: UB
    INTEGER, INTENT(IN) :: DES_PTS_DIM_1
    INTEGER, INTENT(IN) :: DES_PTS_DIM_2
    REAL(KIND=R8), INTENT(IN), DIMENSION(DES_PTS_DIM_1,DES_PTS_DIM_2) :: DES_PTS
    INTEGER, INTENT(IN) :: OBJ_PTS_DIM_1
    INTEGER, INTENT(IN) :: OBJ_PTS_DIM_2
    REAL(KIND=R8), INTENT(IN), DIMENSION(OBJ_PTS_DIM_1,OBJ_PTS_DIM_2) :: OBJ_PTS
    INTEGER, INTENT(IN) :: ISNB
    INTEGER, INTENT(IN) :: SNB
    INTEGER, INTENT(IN) :: ONB
    INTEGER, INTENT(IN) :: OBJ_BOUNDS_DIM_1
    INTEGER, INTENT(IN) :: OBJ_BOUNDS_DIM_2
    REAL(KIND=R8), INTENT(IN), DIMENSION(OBJ_BOUNDS_DIM_1,OBJ_BOUNDS_DIM_2) :: OBJ_BOUNDS
    INTEGER, INTENT(OUT) :: LBATCH
    INTEGER, INTENT(OUT) :: BATCHX_DIM_1
    INTEGER, INTENT(OUT) :: BATCHX_DIM_2
    REAL(KIND=R8), ALLOCATABLE, SAVE, DIMENSION(:,:) :: BATCHX_LOCAL
    INTEGER(KIND=INT64), INTENT(OUT) :: BATCHX
    INTEGER, INTENT(OUT) :: IERR

    CALL VTMOP_LIBE_GENERATE(D, P, LB, UB, DES_PTS, OBJ_PTS, ISNB, SNB, ONB, OBJ_BOUNDS, LBATCH, BATCHX_LOCAL, IERR)

    BATCHX_DIM_1 = SIZE(BATCHX_LOCAL,1)
    BATCHX_DIM_2 = SIZE(BATCHX_LOCAL,2)
    BATCHX = LOC(BATCHX_LOCAL(1,1))
  END SUBROUTINE C_VTMOP_LIBE_GENERATE

END MODULE C_VTMOP_LIBE_MOD