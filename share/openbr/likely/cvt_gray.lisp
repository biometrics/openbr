convert-grayscale :=
  src :->
  {
    (assume src.channels.(== 3))
    dst := (new src.type.not-multi-channel 1 src.columns src.rows src.frames null)
    src := (src.type.not-saturated src)
    (dst src) :=>
      dst :<- (bgr-to-y src.type (src 0) (src 1) (src 2))
  }

(extern u8XY "cvt_gray" u8CXY convert-grayscale)
