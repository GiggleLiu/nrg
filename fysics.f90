!Logarithmic Gaussian broadening for peak.
!NOTE:
    !it is asymmetrix about mean and -mean.

!wlist: a list of w.
!elist: a list of energy.
!b: broadening.
!weights: weights of specific spectrum, it should take the same shape as mean.
 
!return -> an array defined on w-list.
subroutine flog_gaussian(wlist,elist,b,weights,nw,ne,alist)
    implicit none
    integer,intent(in) :: nw,ne
    real*8,intent(in),dimension(nw) :: wlist,b
    real*8,intent(in),dimension(ne) :: elist
    complex*16,intent(in),dimension(ne) :: weights
    complex*16,intent(out),dimension(nw) :: alist

    real*8,parameter :: pi=dacos(-1D0)
    real*8,dimension(ne) :: abselist
    complex*16,dimension(ne) :: weights2
    real*8 :: cb
    integer :: iw
    !f2py intent(in) :: nw,ne,wlist,elist,b,weights
    !f2py intent(out) :: alist

    abselist=abs(elist)+1D-18
    weights2=weights/abselist/sqrt(pi)
    do iw=1,nw
        cb=b(iw)
        alist(iw)=sum(exp(-cb**2/4.)/cb*weights2*exp(-(log(abs(wlist(iw))/abselist)/cb)**2))
    enddo
end subroutine flog_gaussian

!Logarithmic Gaussian broadening for peak, the varied version.
!NOTE:
    !it is asymmetrix about mean and -mean.

!wlist: a list of w.
!elist: a list of energy.
!b: broadening.
!weights: weights of specific spectrum, it should take the same shape as mean.
 
!return -> an array defined on w-list.
subroutine flog_gaussian_var(wlist,elist,b,weights,w0,b0,nw,ne,alist)
    implicit none
    integer,intent(in) :: nw,ne
    real*8,intent(in),dimension(nw) :: wlist
    real*8,intent(in) :: b,w0,b0
    real*8,intent(in),dimension(ne) :: elist
    complex*16,intent(in),dimension(ne) :: weights
    complex*16,intent(out),dimension(nw) :: alist

    real*8,parameter :: pi=dacos(-1D0)
    real*8,dimension(ne) :: abselist
    real*8,dimension(nw) :: h
    complex*16,dimension(ne) :: weights2,weights1
    real*8 :: gm,ee,ww,absee,absww,hi,bb
    complex*16 :: ai
    integer :: iw,ie
    !f2py intent(in) :: nw,ne,wlist,elist,b,weights,w0,b0
    !f2py intent(out) :: alist
    
    !the defult b0
    if(b0<=0) then
        bb=w0*b
    else
        bb=b0
    endif
    gm=b/4
    abselist=abs(elist)+1D-18
    !initialize transition function
    do iw=1,nw
        absww=abs(wlist(iw))
        if(absww>=w0) then
            h(iw)=1
        else
            h(iw)=exp(-(log(absww/w0)/b)**2)
        endif
    enddo
    !weights for gaussian
    weights1=weights/sqrt(pi)/bb
    !weights for log gaussian
    weights2=weights/abselist*(exp(-b**2/4)/sqrt(pi)/b)
    do iw=1,nw
        ww=wlist(iw)
        absww=abs(ww)
        ai=0
        hi=h(iw)
        do ie=1,ne
            ee=elist(ie)
            absee=abselist(ie)
            if(absww<w0) then
                ai=ai+(1-hi)*weights1(ie)*exp(-((ww-elist(ie))/bb)**2)
            endif
            if(ww*ee>=0) then
                ai=ai+hi*weights2(ie)*exp(-(log(abs(wlist(iw))/absee)/b)**2)
            endif
        enddo
        alist(iw)=ai
    enddo
end subroutine flog_gaussian_var

!Logarithmic Gaussian broadening for peak, the varied version.
!NOTE:
    !it is asymmetrix about mean and -mean.

!wlist: a list of w.
!elist: a list of energy.
!b: broadening.
!weights: weights of specific spectrum, it should take the same shape as mean.
 
!return -> an array defined on w-list.
subroutine flog_gaussian_var2(wlist,elist,b,weights,w0,nw,ne,alist)
    implicit none
    integer,intent(in) :: nw,ne
    real*8,intent(in),dimension(nw) :: wlist
    real*8,intent(in) :: b,w0
    real*8,intent(in),dimension(ne) :: elist
    complex*16,intent(in),dimension(ne) :: weights
    complex*16,intent(out),dimension(nw) :: alist

    real*8,parameter :: pi=dacos(-1D0)
    real*8,dimension(ne) :: abselist
    real*8,dimension(ne) :: h
    complex*16,dimension(ne) :: weights2,weights1
    real*8 :: gm,ee,ww,absee,absww,hi
    complex*16 :: ai
    integer :: iw,ie
    !f2py intent(in) :: nw,ne,wlist,elist,b,weights,w0
    !f2py intent(out) :: alist

    gm=b/4
    abselist=abs(elist)+1D-18
    !initialize transition function
    do ie=1,ne
        absee=abselist(ie)
        if(absee>=w0) then
            h(ie)=1
        else
            h(ie)=exp(-(log(absee/w0)/b)**2)
        endif
    enddo
    !weights for gaussian
    weights1=weights/sqrt(pi)/w0
    !weights for log gaussian
    weights2=weights/abselist*(exp(-b*(gm-b/4))/sqrt(pi)/b)
    do iw=1,nw
        ww=wlist(iw)
        absww=abs(ww)
        ai=0
        do ie=1,ne
            hi=h(ie)
            ee=elist(ie)
            absee=abselist(ie)
            if(absee<w0) then
                ai=ai+(1-hi)*weights1(ie)*exp(-((ww-elist(ie))/w0)**2)
            endif
            if(ww*ee>=0) then
                ai=ai+hi*weights2(ie)*exp(-(log(abs(ww)/absee)/b+gm-b/2)**2)
            endif
        enddo
        alist(iw)=ai
    enddo
end subroutine flog_gaussian_var2

