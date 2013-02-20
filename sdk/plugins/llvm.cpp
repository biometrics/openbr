#include <QMetaProperty>
#include <llvm/Intrinsics.h>
#include <llvm/LLVMContext.h>
#include <llvm/DerivedTypes.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/IRBuilder.h>
#include <llvm/Module.h>
#include <llvm/Operator.h>
#include <llvm/PassManager.h>
#include <llvm/Type.h>
#include <llvm/Value.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/Assembly/PrintModulePass.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Scalar.h>
#include <openbr_plugin.h>
#include <likely.h>

#include "core/opencvutils.h"

using namespace br;
using namespace cv;
using namespace llvm;

typedef uint32_t (*likely_unary_allocation)(const likely_matrix *src, likely_matrix *dst);
typedef uint32_t (*likely_binary_allocation)(const likely_matrix *srcA, const likely_matrix *srcB, likely_matrix *dst);
typedef void (*likely_unary_kernel)(const likely_matrix *src, likely_matrix *dst, uint32_t size);
typedef void (*likely_binary_kernel)(const likely_matrix *srcA, const likely_matrix *srcB, likely_matrix *dst, uint32_t size);

static Module *TheModule = NULL;
static ExecutionEngine *TheExecutionEngine = NULL;
static FunctionPassManager *TheFunctionPassManager = NULL;
static FunctionPassManager *TheExtraFunctionPassManager = NULL;
static StructType *TheMatrixStruct = NULL;

static void likely_initialize()
{
    InitializeNativeTarget();

    TheModule = new Module("likely", getGlobalContext());

    std::string error;
    TheExecutionEngine = EngineBuilder(TheModule).setEngineKind(EngineKind::JIT).setErrorStr(&error).create();
    if (TheExecutionEngine == NULL)
        qFatal("Failed to create LLVM ExecutionEngine with error: %s", error.c_str());

    TheFunctionPassManager = new FunctionPassManager(TheModule);
    TheFunctionPassManager->add(createVerifierPass(PrintMessageAction));
    TheFunctionPassManager->add(createEarlyCSEPass());
    TheFunctionPassManager->add(createInstructionCombiningPass());
    TheFunctionPassManager->add(createDeadCodeEliminationPass());
    TheFunctionPassManager->add(createGVNPass());
    TheFunctionPassManager->add(createDeadInstEliminationPass());

    TheExtraFunctionPassManager = new FunctionPassManager(TheModule);
    TheExtraFunctionPassManager->add(createPrintFunctionPass("--------------------------------------------------------------------------------", &errs()));
//        TheExtraFunctionPassManager->add(createLoopUnrollPass(INT_MAX,8));

    TheMatrixStruct = StructType::create("Matrix",
                                         Type::getInt8PtrTy(getGlobalContext()), // data
                                         Type::getInt32Ty(getGlobalContext()),   // channels
                                         Type::getInt32Ty(getGlobalContext()),   // columns
                                         Type::getInt32Ty(getGlobalContext()),   // rows
                                         Type::getInt32Ty(getGlobalContext()),   // frames
                                         Type::getInt16Ty(getGlobalContext()),   // hash
                                         NULL);
}

static QString MatrixToString(const likely_matrix *m)
{
    return QString("%1%2%3%4%5%6%7").arg(QString::number(likely_bits(m)), (likely_is_signed(m) ? "s" : "u"), (likely_is_floating(m) ? "f" : "i"),
                                         QString::number(likely_is_single_channel(m)), QString::number(likely_is_single_column(m)), QString::number(likely_is_single_row(m)), QString::number(likely_is_single_frame(m)));
}

static likely_matrix MatrixFromMat(const cv::Mat &mat)
{
    likely_matrix m;

    if (!mat.isContinuous()) qFatal("Matrix requires continuous data.");
    m.channels = mat.channels();
    m.columns = mat.cols;
    m.rows = mat.rows;
    m.frames = 1;

    switch (mat.depth()) {
      case CV_8U:  m.hash = likely_matrix::u8;  break;
      case CV_8S:  m.hash = likely_matrix::s8;  break;
      case CV_16U: m.hash = likely_matrix::u16; break;
      case CV_16S: m.hash = likely_matrix::s16; break;
      case CV_32S: m.hash = likely_matrix::s32; break;
      case CV_32F: m.hash = likely_matrix::f32; break;
      case CV_64F: m.hash = likely_matrix::f64; break;
      default:     qFatal("Unrecognized matrix depth.");
    }

    m.data = mat.data;
    return m;
}

static Mat MatFromMatrix(const likely_matrix &m)
{
    int depth = -1;
    switch (likely_type(&m)) {
      case likely_matrix::u8:  depth = CV_8U;  break;
      case likely_matrix::s8:  depth = CV_8S;  break;
      case likely_matrix::u16: depth = CV_16U; break;
      case likely_matrix::s16: depth = CV_16S; break;
      case likely_matrix::s32: depth = CV_32S; break;
      case likely_matrix::f32: depth = CV_32F; break;
      case likely_matrix::f64: depth = CV_64F; break;
      default:     qFatal("Unrecognized matrix depth.");
    }
    return Mat(m.rows, m.columns, CV_MAKETYPE(depth, m.channels), m.data).clone();
}

QDebug operator<<(QDebug dbg, const likely_matrix &m)
{
    dbg.nospace() << MatrixToString(&m);
    return dbg;
}

struct MatrixBuilder
{
    const likely_matrix *m;
    Value *v;
    IRBuilder<> *b;
    Function *f;
    Twine name;

    MatrixBuilder(const likely_matrix *matrix, Value *value, IRBuilder<> *builder, Function *function, const Twine &name_)
        : m(matrix), v(value), b(builder), f(function), name(name_) {}

    static Constant *constant(int value, int bits = 32) { return Constant::getIntegerValue(Type::getInt32Ty(getGlobalContext()), APInt(bits, value)); }
    static Constant *constant(float value) { return ConstantFP::get(Type::getFloatTy(getGlobalContext()), value == 0 ? -0.0f : value); }
    static Constant *constant(double value) { return ConstantFP::get(Type::getDoubleTy(getGlobalContext()), value == 0 ? -0.0 : value); }
    static Constant *zero() { return constant(0); }
    static Constant *one() { return constant(1); }
    Constant *autoConstant(double value) const { return ::likely_is_floating(m) ? ((::likely_bits(m) == 64) ? constant(value) : constant(float(value))) : constant(int(value), ::likely_bits(m)); }
    AllocaInst *autoAlloca(double value, const Twine &name = "") const { AllocaInst *alloca = b->CreateAlloca(ty(), 0, name); b->CreateStore(autoConstant(value), alloca); return alloca; }

    Value *data(Value *matrix, const Twine &name = "") const { return b->CreateLoad(b->CreateStructGEP(matrix, 0), name+"_data"); }
    Value *data(Value *matrix, Type *type, const Twine &name = "") const { return b->CreatePointerCast(data(matrix, name), type); }
    Value *channels(Value *matrix, const Twine &name = "") const { return b->CreateLoad(b->CreateStructGEP(matrix, 1), name+"_channels"); }
    Value *columns(Value *matrix, const Twine &name = "") const { return b->CreateLoad(b->CreateStructGEP(matrix, 2), name+"_columns"); }
    Value *rows(Value *matrix, const Twine &name = "") const { return b->CreateLoad(b->CreateStructGEP(matrix, 3), name+"_rows"); }
    Value *frames(Value *matrix, const Twine &name = "") const { return b->CreateLoad(b->CreateStructGEP(matrix, 4), name+"_frames"); }
    Value *hash(Value *matrix, const Twine &name = "") const { return b->CreateLoad(b->CreateStructGEP(matrix, 5), name+"_hash"); }

    Value *data(bool cast = true) const { return cast ? data(v, ptrTy(), name) : data(v, name); }
    Value *channels() const { return ::likely_is_single_channel(m) ? static_cast<Value*>(one()) : channels(v, name); }
    Value *columns() const { return ::likely_is_single_column(m) ? static_cast<Value*>(one()) : columns(v, name); }
    Value *rows() const { return ::likely_is_single_row(m) ? static_cast<Value*>(one()) : rows(v, name); }
    Value *frames() const { return ::likely_is_single_frame(m) ? static_cast<Value*>(one()) : frames(v, name); }
    Value *hash() const { return hash(v, name); }

    void setData(Value *matrix, Value *value) const { b->CreateStore(value, b->CreateStructGEP(matrix, 0)); }
    void setChannels(Value *matrix, Value *value) const { b->CreateStore(value, b->CreateStructGEP(matrix, 1)); }
    void setColumns(Value *matrix, Value *value) const { b->CreateStore(value, b->CreateStructGEP(matrix, 2)); }
    void setRows(Value *matrix, Value *value) const { b->CreateStore(value, b->CreateStructGEP(matrix, 3)); }
    void setFrames(Value *matrix, Value *value) const { b->CreateStore(value, b->CreateStructGEP(matrix, 4)); }
    void setHash(Value *matrix, Value *value) const { b->CreateStore(value, b->CreateStructGEP(matrix, 5)); }

    void setData(Value *value) const { setData(v, value); }
    void setChannels(Value *value) const { setChannels(v, value); }
    void setColumns(Value *value) const { setColumns(v, value); }
    void setRows(Value *value) const { setRows(v, value); }
    void setFrames(Value *value) const { setFrames(v, value); }
    void setHash(Value *value) const { setHash(v, value); }

    void copyHeaderTo(Value *matrix) const {
        setChannels(matrix, channels());
        setColumns(matrix, columns());
        setRows(matrix, rows());
        setFrames(matrix, frames());
        setHash(matrix, hash());
    }

    void deallocate() const {
        static Function *free = TheModule->getFunction("free");
        if (!free) {
            Type *freeReturn = Type::getVoidTy(getGlobalContext());
            std::vector<Type*> freeParams;
            freeParams.push_back(Type::getInt8PtrTy(getGlobalContext()));
            FunctionType* freeType = FunctionType::get(freeReturn, freeParams, false);
            free = Function::Create(freeType, GlobalValue::ExternalLinkage, "free", TheModule);
            free->setCallingConv(CallingConv::C);
        }

        std::vector<Value*> freeArgs;
        freeArgs.push_back(b->CreateStructGEP(v, 0));
        b->CreateCall(free, freeArgs);
        setData(ConstantPointerNull::get(Type::getInt8PtrTy(getGlobalContext())));
    }

    Value *get(int mask) const { return b->CreateAnd(hash(), constant(mask, 16)); }
    void set(int value, int mask) const { setHash(b->CreateOr(b->CreateAnd(hash(), constant(~mask, 16)), b->CreateAnd(constant(value, 16), constant(mask, 16)))); }
    void setBit(bool on, int mask) const { on ? setHash(b->CreateOr(hash(), constant(mask, 16))) : setHash(b->CreateAnd(hash(), constant(~mask, 16))); }

    Value *bits() const { return get(likely_matrix::Bits); }
    void setBits(int bits) const { set(bits, likely_matrix::Bits); }
    Value *isFloating() const { return get(likely_matrix::Floating); }
    void setFloating(bool isFloating) const { if (isFloating) setSigned(true); setBit(isFloating, likely_matrix::Floating); }
    Value *isSigned() const { return get(likely_matrix::Signed); }
    void setSigned(bool isSigned) const { setBit(isSigned, likely_matrix::Signed); }
    Value *type() const { return get(likely_matrix::Bits + likely_matrix::Floating + likely_matrix::Signed); }
    void setType(int type) const { set(type, likely_matrix::Bits + likely_matrix::Floating + likely_matrix::Signed); }
    Value *singleChannel() const { return get(likely_matrix::SingleChannel); }
    void setSingleChannel(bool singleChannel) const { setBit(singleChannel, likely_matrix::SingleChannel); }
    Value *singleColumn() const { return get(likely_matrix::SingleColumn); }
    void setSingleColumn(bool singleColumn) { setBit(singleColumn, likely_matrix::SingleColumn); }
    Value *singleRow() const { return get(likely_matrix::SingleRow); }
    void setSingleRow(bool singleRow) const { setBit(singleRow, likely_matrix::SingleRow); }
    Value *singleFrame() const { return get(likely_matrix::SingleFrame); }
    void setSingleFrame(bool singleFrame) const { setBit(singleFrame, likely_matrix::SingleFrame); }
    Value *elements() const { return b->CreateMul(b->CreateMul(b->CreateMul(channels(), columns()), rows()), frames()); }
    Value *bytes() const { return b->CreateMul(b->CreateUDiv(b->CreateCast(Instruction::ZExt, bits(), Type::getInt32Ty(getGlobalContext())), constant(8, 32)), elements()); }

    Value *columnStep() const { Value *columnStep = channels(); columnStep->setName(name+"_cStep"); return columnStep; }
    Value *rowStep() const { return b->CreateMul(columns(), columnStep(), name+"_rStep"); }
    Value *frameStep() const { return b->CreateMul(rows(), rowStep(), name+"_tStep"); }
    Value *aliasColumnStep(const MatrixBuilder &other) const { return (m->channels == other.m->channels) ? other.columnStep() : columnStep(); }
    Value *aliasRowStep(const MatrixBuilder &other) const { return (m->columns == other.m->columns) ? other.rowStep() : rowStep(); }
    Value *aliasFrameStep(const MatrixBuilder &other) const { return (m->rows == other.m->rows) ? other.frameStep() : frameStep(); }

    Value *index(Value *c) const { return ::likely_is_single_channel(m) ? constant(0) : c; }
    Value *index(Value *c, Value *x) const { return ::likely_is_single_column(m) ? index(c) : b->CreateAdd(b->CreateMul(x, columnStep()), index(c)); }
    Value *index(Value *c, Value *x, Value *y) const { return ::likely_is_single_row(m) ? index(c, x) : b->CreateAdd(b->CreateMul(y, rowStep()), index(c, x)); }
    Value *index(Value *c, Value *x, Value *y, Value *f) const { return ::likely_is_single_frame(m) ? index(c, x, y) : b->CreateAdd(b->CreateMul(f, frameStep()), index(c, x, y)); }
    Value *aliasIndex(const MatrixBuilder &other, Value *c, Value *x) const { return ::likely_is_single_column(m) ? index(c) : b->CreateAdd(b->CreateMul(x, aliasColumnStep(other)), index(c)); }
    Value *aliasIndex(const MatrixBuilder &other, Value *c, Value *x, Value *y) const { return ::likely_is_single_row(m) ? aliasIndex(other, c, x) : b->CreateAdd(b->CreateMul(y, aliasRowStep(other)), aliasIndex(other, c, x)); }
    Value *aliasIndex(const MatrixBuilder &other, Value *c, Value *x, Value *y, Value *f) const { return ::likely_is_single_frame(m) ? aliasIndex(other, c, x, y) : b->CreateAdd(b->CreateMul(f, aliasFrameStep(other)), aliasIndex(other, c, x, y)); }

    void deindex(Value *i, Value **c) const {
        *c = ::likely_is_single_channel(m) ? constant(0) : i;
    }
    void deindex(Value *i, Value **c, Value **x) const {
        Value *rem;
        if (::likely_is_single_column(m)) {
            rem = i;
            *x = constant(0);
        } else {
            Value *step = columnStep();
            rem = b->CreateURem(i, step, name+"_xRem");
            *x = b->CreateExactUDiv(b->CreateSub(i, rem), step, name+"_x");
        }
        deindex(rem, c);
    }
    void deindex(Value *i, Value **c, Value **x, Value **y) const {
        Value *rem;
        if (::likely_is_single_row(m)) {
            rem = i;
            *y = constant(0);
        } else {
            Value *step = rowStep();
            rem = b->CreateURem(i, step, name+"_yRem");
            *y = b->CreateExactUDiv(b->CreateSub(i, rem), step, name+"_y");
        }
        deindex(rem, c, x);
    }
    void deindex(Value *i, Value **c, Value **x, Value **y, Value **t) const {
        Value *rem;
        if (::likely_is_single_frame(m)) {
            rem = i;
            *t = constant(0);
        } else {
            Value *step = frameStep();
            rem = b->CreateURem(i, step, name+"_tRem");
            *t = b->CreateExactUDiv(b->CreateSub(i, rem), step, name+"_t");
        }
        deindex(rem, c, x, y);
    }

    LoadInst *load(Value *matrix, Type *type, Value *i) const { return b->CreateLoad(b->CreateGEP(data(matrix, type), i)); }
    LoadInst *load(Value *i) const { return b->CreateLoad(b->CreateGEP(data(), i)); }
    StoreInst *store(Value *matrix, Type *type, Value *i, Value *value) const { Value *d = data(matrix, type);
                                                                                d->dump(); i->dump();
                                                                                Value *idx = b->CreateGEP(d, i);
                                                                                return b->CreateStore(value, idx); }
    StoreInst *store(Value *i, Value *value) const { return b->CreateStore(value, b->CreateGEP(data(), i)); }

    Value *cast(Value *i, const MatrixBuilder &dst) const { return (::likely_type(m) == ::likely_type(dst.m)) ? i : b->CreateCast(CastInst::getCastOpcode(i, ::likely_is_signed(m), dst.ty(), ::likely_is_signed(dst.m)), i, dst.ty()); }
    Value *add(Value *i, Value *j, const Twine &name = "") const { return ::likely_is_floating(m) ? b->CreateFAdd(i, j, name) : b->CreateAdd(i, j, name); }
    Value *multiply(Value *i, Value *j, const Twine &name = "") const { return ::likely_is_floating(m) ? b->CreateFMul(i, j, name) : b->CreateMul(i, j, name); }

    Value *compareLT(Value *i, Value *j) const { return ::likely_is_floating(m) ? b->CreateFCmpOLT(i, j) : (::likely_is_signed(m) ? b->CreateICmpSLT(i, j) : b->CreateICmpULT(i, j)); }
    Value *compareGT(Value *i, Value *j) const { return ::likely_is_floating(m) ? b->CreateFCmpOGT(i, j) : (::likely_is_signed(m) ? b->CreateICmpSGT(i, j) : b->CreateICmpUGT(i, j)); }

    static PHINode *beginLoop(IRBuilder<> &builder, Function *function, BasicBlock *entry, BasicBlock *&loop, BasicBlock *&exit, Value *stop, const Twine &name = "") {
        loop = BasicBlock::Create(getGlobalContext(), "loop_"+name, function);
        builder.CreateBr(loop);
        builder.SetInsertPoint(loop);

        PHINode *i = builder.CreatePHI(Type::getInt32Ty(getGlobalContext()), 2, name);
        i->addIncoming(MatrixBuilder::zero(), entry);
        Value *increment = builder.CreateAdd(i, MatrixBuilder::one(), "increment_"+name);
        BasicBlock *body = BasicBlock::Create(getGlobalContext(), "loop_"+name+"_body", function);
        i->addIncoming(increment, body);

        exit = BasicBlock::Create(getGlobalContext(), "loop_"+name+"_exit", function);
        builder.CreateCondBr(builder.CreateICmpEQ(i, stop, "loop_"+name+"_test"), exit, body);
        builder.SetInsertPoint(body);
        return i;
    }
    PHINode *beginLoop(BasicBlock *entry, BasicBlock *&loop, BasicBlock *&exit, Value *stop, const Twine &name = "") const { return beginLoop(*b, f, entry, loop, exit, stop, name); }

    static void endLoop(IRBuilder<> &builder, BasicBlock *loop, BasicBlock *exit) {
        builder.CreateBr(loop);
        builder.SetInsertPoint(exit);
    }
    void endLoop(BasicBlock *loop, BasicBlock *exit) const { endLoop(*b, loop, exit); }

    template <typename T>
    inline static std::vector<T> toVector(T value) { std::vector<T> vector; vector.push_back(value); return vector; }

    static Type *ty(const likely_matrix &m)
    {
        const int bits = ::likely_bits(&m);
        if (::likely_is_floating(&m)) {
            if      (bits == 16) return Type::getHalfTy(getGlobalContext());
            else if (bits == 32) return Type::getFloatTy(getGlobalContext());
            else if (bits == 64) return Type::getDoubleTy(getGlobalContext());
        } else {
            if      (bits == 1)  return Type::getInt1Ty(getGlobalContext());
            else if (bits == 8)  return Type::getInt8Ty(getGlobalContext());
            else if (bits == 16) return Type::getInt16Ty(getGlobalContext());
            else if (bits == 32) return Type::getInt32Ty(getGlobalContext());
            else if (bits == 64) return Type::getInt64Ty(getGlobalContext());
        }
        qFatal("Invalid matrix type.");
        return NULL;
    }
    inline Type *ty() const { return ty(*m); }
    inline std::vector<Type*> tys() const { return toVector<Type*>(ty()); }

    static Type *ptrTy(const likely_matrix &m)
    {
        const int bits = ::likely_bits(&m);
        if (::likely_is_floating(&m)) {
            if      (bits == 16) return Type::getHalfPtrTy(getGlobalContext());
            else if (bits == 32) return Type::getFloatPtrTy(getGlobalContext());
            else if (bits == 64) return Type::getDoublePtrTy(getGlobalContext());
        } else {
            if      (bits == 1)  return Type::getInt1PtrTy(getGlobalContext());
            else if (bits == 8)  return Type::getInt8PtrTy(getGlobalContext());
            else if (bits == 16) return Type::getInt16PtrTy(getGlobalContext());
            else if (bits == 32) return Type::getInt32PtrTy(getGlobalContext());
            else if (bits == 64) return Type::getInt64PtrTy(getGlobalContext());
        }
        qFatal("Invalid matrix type.");
        return NULL;
    }
    inline Type *ptrTy() const { return ptrTy(*m); }
};

namespace br
{

/*!
 * \brief LLVM Unary Transform
 * \author Josh Klontz \cite jklontz
 */
class UnaryTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    uint32_t fileIndex;

public:
    static QHash<uint32_t, File> fileTable;

    UnaryTransform() : fileIndex(0) {}

    void init()
    {
        fileIndex = fileTable.size()+1;
        fileTable.insert(fileIndex, file);
    }

    virtual Value *preallocation(const MatrixBuilder &src, Value *dst) const = 0; /*!< Allocate the destintation matrix given the source matrix. */
    virtual void kernel(const MatrixBuilder &src, Value *dst, PHINode *i) const = 0; /*!< Run the computation given the source matrix. */

    void optimize(Function *f) const
    {
        while (TheFunctionPassManager->run(*f));
        TheExtraFunctionPassManager->run(*f);
    }

    likely_unary_function getFunction() const
    {
        if (TheModule == NULL) likely_initialize();

        const QString name = mangledName();
        Function *function = TheModule->getFunction(qPrintable(name));

        if (function == NULL) {
            static Function *makeAllocationFunction = NULL;
            static PointerType *allocationType = NULL;
            if (makeAllocationFunction == NULL) {
                std::vector<Type*> allocationParams;
                allocationParams.push_back(PointerType::getUnqual(TheMatrixStruct));
                allocationParams.push_back(PointerType::getUnqual(TheMatrixStruct));
                Type *allocationReturn = Type::getInt32Ty(getGlobalContext());
                allocationType = PointerType::getUnqual(FunctionType::get(allocationReturn, allocationParams, false));
                std::vector<Type*> makeAllocationParams;
                makeAllocationParams.push_back(Type::getInt8PtrTy(getGlobalContext()));
                makeAllocationParams.push_back(PointerType::getUnqual(TheMatrixStruct));
                FunctionType* makeAllocationType = FunctionType::get(allocationType, makeAllocationParams, false);
                makeAllocationFunction = Function::Create(makeAllocationType, GlobalValue::ExternalLinkage, "likely_make_unary_allocation", TheModule);
                makeAllocationFunction->setCallingConv(CallingConv::C);
            }

            static Function *makeKernelFunction = NULL;
            static PointerType *kernelType = NULL;
            if (makeKernelFunction == NULL) {
                std::vector<Type*> kernelParams;
                kernelParams.push_back(PointerType::getUnqual(TheMatrixStruct));
                kernelParams.push_back(PointerType::getUnqual(TheMatrixStruct));
                kernelParams.push_back(Type::getInt32Ty(getGlobalContext()));
                Type *kernelReturn = Type::getVoidTy(getGlobalContext());
                kernelType = PointerType::getUnqual(FunctionType::get(kernelReturn, kernelParams, false));
                std::vector<Type*> makeKernelParams;
                makeKernelParams.push_back(Type::getInt8PtrTy(getGlobalContext()));
                makeKernelParams.push_back(PointerType::getUnqual(TheMatrixStruct));
                FunctionType* makeUnaryKernelType = FunctionType::get(kernelType, makeKernelParams, false);
                makeKernelFunction = Function::Create(makeUnaryKernelType, GlobalValue::ExternalLinkage, "likely_make_unary_kernel", TheModule);
                makeKernelFunction->setCallingConv(CallingConv::C);
            }

            function = cast<Function>(TheModule->getOrInsertFunction(qPrintable(name),
                                                                     Type::getVoidTy(getGlobalContext()),
                                                                     PointerType::getUnqual(TheMatrixStruct),
                                                                     PointerType::getUnqual(TheMatrixStruct),
                                                                     NULL));
            function->setCallingConv(CallingConv::C);

            Function::arg_iterator args = function->arg_begin();
            Value *src = args++;
            src->setName("src");
            Value *dst = args++;
            dst->setName("dst");

            BasicBlock *entry = BasicBlock::Create(getGlobalContext(), "entry", function);
            IRBuilder<> builder(entry);

            GlobalVariable *kernelHash; {
                kernelHash = cast<GlobalVariable>(TheModule->getOrInsertGlobal(qPrintable(name+"_hash"), Type::getInt16Ty(getGlobalContext())));
                kernelHash->setInitializer(MatrixBuilder::constant(0, 16));
            }

            GlobalVariable *allocationFunction; {
                allocationFunction = cast<GlobalVariable>(TheModule->getOrInsertGlobal(qPrintable(name+"_allocation"), allocationType));
                allocationFunction->setInitializer(ConstantPointerNull::get(allocationType));
            }

            GlobalVariable *kernelFunction; {
                kernelFunction = cast<GlobalVariable>(TheModule->getOrInsertGlobal(qPrintable(name+"_kernel"), kernelType));
                kernelFunction->setInitializer(ConstantPointerNull::get(kernelType));
            }

            BasicBlock *hashFail = BasicBlock::Create(getGlobalContext(), "hash_fail", function);
            BasicBlock *execute = BasicBlock::Create(getGlobalContext(), "execute", function);
            Value *srcHash = builder.CreateLoad(builder.CreateStructGEP(src, 5), "src_hash");
            Value *hashTest = builder.CreateICmpEQ(srcHash, builder.CreateLoad(kernelHash), "hash_fail_test");
            builder.CreateCondBr(hashTest, execute, hashFail);

            builder.SetInsertPoint(hashFail);
            builder.CreateStore(builder.CreateCall2(makeAllocationFunction,
                                                    builder.CreateIntToPtr(MatrixBuilder::constant(fileIndex, 32), Type::getInt8PtrTy(getGlobalContext())),
                                                    src), allocationFunction);
            builder.CreateStore(builder.CreateCall2(makeKernelFunction,
                                                    builder.CreateIntToPtr(MatrixBuilder::constant(fileIndex, 32), Type::getInt8PtrTy(getGlobalContext())),
                                                    src), kernelFunction);
            builder.CreateStore(srcHash, kernelHash);
            builder.CreateBr(execute);

            builder.SetInsertPoint(execute);
            Value *kernelSize = builder.CreateCall2(builder.CreateLoad(allocationFunction), src, dst);
            builder.CreateCall3(builder.CreateLoad(kernelFunction), src, dst, kernelSize);
            builder.CreateRetVoid();

            optimize(function);
        }

        return (likely_unary_function)TheExecutionEngine->getPointerToFunction(function);
    }

    likely_unary_allocation getAllocation(const likely_matrix *m) const
    {
        const QString name = mangledName(*m)+"_allocation";
        Function *function = TheModule->getFunction(qPrintable(name));

        if (function == NULL) {
            static Function *malloc = TheModule->getFunction("malloc");
            if (malloc == NULL) {
                Type *mallocReturn = Type::getInt8PtrTy(getGlobalContext());
                std::vector<Type*> mallocParams;
                mallocParams.push_back(Type::getInt32Ty(getGlobalContext()));
                FunctionType* mallocType = FunctionType::get(mallocReturn, mallocParams, false);
                malloc = Function::Create(mallocType, GlobalValue::ExternalLinkage, "malloc", TheModule);
                malloc->setCallingConv(CallingConv::C);
            }

            function = cast<Function>(TheModule->getOrInsertFunction(qPrintable(name),
                                                                     Type::getInt32Ty(getGlobalContext()),
                                                                     PointerType::getUnqual(TheMatrixStruct),
                                                                     PointerType::getUnqual(TheMatrixStruct),
                                                                     NULL));
            function->setCallingConv(CallingConv::C);

            Function::arg_iterator args = function->arg_begin();
            Value *src = args++;
            src->setName("src");
            Value *dst = args++;
            dst->setName("dst");

            BasicBlock *entry = BasicBlock::Create(getGlobalContext(), "entry", function);
            IRBuilder<> builder(entry);
            MatrixBuilder mb(m, src, &builder, function, "src");

            Value *kernelSize = preallocation(mb, dst);
            mb.setData(dst, builder.CreateCall(malloc, kernelSize));
            builder.CreateRet(kernelSize);

            optimize(function);
        }

        return (likely_unary_allocation)TheExecutionEngine->getPointerToFunction(function);
    }

    likely_unary_kernel getKernel(const likely_matrix *m) const
    {
        const QString name = mangledName(*m)+"_kernel";
        Function *function = TheModule->getFunction(qPrintable(name));

        if (function == NULL) {
            function = cast<Function>(TheModule->getOrInsertFunction(qPrintable(name),
                                                                     Type::getVoidTy(getGlobalContext()),
                                                                     PointerType::getUnqual(TheMatrixStruct),
                                                                     PointerType::getUnqual(TheMatrixStruct),
                                                                     Type::getInt32Ty(getGlobalContext()),
                                                                     NULL));
            function->setCallingConv(CallingConv::C);

            Function::arg_iterator args = function->arg_begin();
            Value *src = args++;
            src->setName("src");
            Value *dst = args++;
            dst->setName("dst");
            Value *len = args++;
            len->setName("len");

            BasicBlock *entry = BasicBlock::Create(getGlobalContext(), "entry", function);
            IRBuilder<> builder(entry);

            BasicBlock *loop, *exit;
            PHINode *i = MatrixBuilder::beginLoop(builder, function, entry, loop, exit, len, "i");

            kernel(MatrixBuilder(m, src, &builder, function, "src"), dst, i);

            MatrixBuilder::endLoop(builder, loop, exit);
            builder.CreateRetVoid();

            optimize(function);
        }

        return (likely_unary_kernel)TheExecutionEngine->getPointerToFunction(function);
    }

private:
    QString mangledName() const
    {
        static QHash<QString, int> argsLUT;
        const QString args = arguments().join(",");
        if (!argsLUT.contains(args)) argsLUT.insert(args, argsLUT.size());
        int uid = argsLUT.value(args);
        return "likely_" + objectName() + (args.isEmpty() ? QString() : QString::number(uid));
    }

    QString mangledName(const likely_matrix &src) const
    {
        return mangledName() + "_" + MatrixToString(&src);
    }

    void project(const Template &src, Template &dst) const
    {
        const likely_matrix m(MatrixFromMat(src));
        likely_matrix n;
        likely_unary_function function = getFunction();
        function(&m, &n);
        dst.m() = MatFromMatrix(n);
    }
};

QHash<uint32_t, File> UnaryTransform::fileTable;

/*!
 * \brief LLVM Binary Kernel
 * \author Josh Klontz \cite jklontz
 */
//class BinaryTransform: public UntrainableMetaTransform
//{
//    Q_OBJECT

//    BinaryKernel kernel;
//    quint16 hashA, hashB;

//public:
//    BinaryTransform() : kernel(NULL), hashA(0), hashB(0) {}
//    virtual int preallocate(const Matrix &srcA, const Matrix &srcB, Matrix &dst) const = 0; /*!< Preallocate destintation matrix based on source matrix. */
//    virtual void build(const MatrixBuilder &srcA, const MatrixBuilder &srcB, const MatrixBuilder &dst, PHINode *i) const = 0; /*!< Build the kernel. */

//    void apply(const Matrix &srcA, const Matrix &srcB, Matrix &dst) const
//    {
//        const int size = preallocate(srcA, srcB, dst);
//        dst.allocate();
//        invoke(srcA, srcB, dst, size);
//    }

//private:
//    QString mangledName(const Matrix &srcA, const Matrix &srcB) const
//    {
//        return "likely_" + objectName() + "_" + MatrixToString(srcA) + "_" + MatrixToString(srcB);
//    }

//    Function *compile(const Matrix &m, const Matrix &n) const
//    {
//        Constant *c = TheModule->getOrInsertFunction(qPrintable(mangledName(m, n)),
//                                                     Type::getVoidTy(getGlobalContext()),
//                                                     PointerType::getUnqual(TheMatrixStruct),
//                                                     PointerType::getUnqual(TheMatrixStruct),
//                                                     PointerType::getUnqual(TheMatrixStruct),
//                                                     Type::getInt32Ty(getGlobalContext()),
//                                                     NULL);

//        Function *function = cast<Function>(c);
//        function->setCallingConv(CallingConv::C);

//        Function::arg_iterator args = function->arg_begin();
//        Value *srcA = args++;
//        srcA->setName("srcA");
//        Value *srcB = args++;
//        srcB->setName("srcB");
//        Value *dst = args++;
//        dst->setName("dst");
//        Value *len = args++;
//        len->setName("len");

//        BasicBlock *entry = BasicBlock::Create(getGlobalContext(), "entry", function);
//        IRBuilder<> builder(entry);

//        BasicBlock *loop, *exit;
//        PHINode *i = MatrixBuilder::beginLoop(builder, function, entry, loop, exit, len, "i");

//        Matrix o;
//        preallocate(m, n, o);
//        build(MatrixBuilder(&m, srcA, &builder, function, "srcA"), MatrixBuilder(&n, srcB, &builder, function, "srcB"), MatrixBuilder(&o, dst, &builder, function, "dst"), i);

//        MatrixBuilder::endLoop(builder, loop, exit);

//        builder.CreateRetVoid();
//        return function;
//    }

//    void invoke(const Matrix &srcA, const Matrix &srcB, Matrix &dst, int size) const
//    {
//        if ((srcA.hash != hashA) || (srcB.hash != hashB)) {
//            static QMutex compilerLock;
//            QMutexLocker locker(&compilerLock);

//            if ((srcA.hash != hashA) || (srcB.hash != hashB)) {
//                const QString functionName = mangledName(srcA, srcB);

//                Function *function = TheModule->getFunction(qPrintable(functionName));
//                if (function == NULL) {
//                    function = compile(srcA, srcB);
//                    while (TheFunctionPassManager->run(*function));
//                    TheExtraFunctionPassManager->run(*function);
//                    function = TheModule->getFunction(qPrintable(functionName));
//                }

//                const_cast<BinaryTransform*>(this)->kernel = (BinaryKernel)TheExecutionEngine->getPointerToFunction(function);
//                const_cast<BinaryTransform*>(this)->hashA = srcA.hash;
//                const_cast<BinaryTransform*>(this)->hashB = srcB.hash;
//            }
//        }

//        kernel(&srcA, &srcB, &dst, size);
//    }
//};

/*!
 * \brief LLVM Stitchable Kernel
 * \author Josh Klontz \cite jklontz
 */
class StitchableTransform : public UnaryTransform
{
    Q_OBJECT

public:
    virtual Value *stitch(const MatrixBuilder &src, Value *val) const = 0; /*!< A simplification of Kernel::build() for stitchable kernels. */

    virtual Value *preallocation(const MatrixBuilder &src, Value *dst) const
    {
        src.copyHeaderTo(dst);
        return src.elements();
    }

private:
    void kernel(const MatrixBuilder &src, Value *dst, PHINode *i) const
    {
        src.store(dst, src.ptrTy(), i, stitch(src, src.load(i)));
    }
};

} // namespace br

/*!
 * \ingroup transforms
 * \brief LLVM stitch transform
 * \author Josh Klontz \cite jklontz
 */
//class stitchTransform : public UnaryTransform
//{
//    Q_OBJECT
//    Q_PROPERTY(QList<br::Transform*> kernels READ get_kernels WRITE set_kernels RESET reset_kernels STORED false)
//    BR_PROPERTY(QList<br::Transform*>, kernels, QList<br::Transform*>())

//    void init()
//    {
//        foreach (Transform *transform, kernels)
//            if (dynamic_cast<StitchableTransform*>(transform) == NULL)
//                qFatal("%s is not a stitchable transform!", qPrintable(transform->objectName()));
//    }

//    int preallocate(const Matrix &src, Matrix &dst) const
//    {
//        Matrix tmp = src;
//        foreach (const Transform *kernel, kernels) {
//            static_cast<const UnaryTransform*>(kernel)->preallocate(tmp, dst);
//            tmp = dst;
//        }
//        return dst.elements();
//    }

//    void build(const MatrixBuilder &src_, const MatrixBuilder &dst_, PHINode *i) const
//    {
//        MatrixBuilder src(src_);
//        MatrixBuilder dst(dst_);
//        Value *val = src.load(i);
//        foreach (Transform *transform, kernels) {
//            static_cast<UnaryTransform*>(transform)->preallocate(*src.m, *const_cast<Matrix*>(dst.m));
//            val = static_cast<StitchableTransform*>(transform)->stitch(src, dst, val);
//            //src.m->copyHeader(*dst.m);
//            //src.v = dst.v;
//        }
//        dst.store(i, val);
//    }
//};

//BR_REGISTER(Transform, stitchTransform)

/*!
 * \ingroup transforms
 * \brief LLVM square transform
 * \author Josh Klontz \cite jklontz
 */
//class squareTransform : public StitchableTransform
//{
//    Q_OBJECT

//    Value *stitch(const MatrixBuilder &src, const MatrixBuilder &dst, Value *val) const
//    {
//        (void) src;
//        return dst.multiply(val, val);
//    }
//};

//BR_REGISTER(Transform, squareTransform)

/*!
 * \ingroup transforms
 * \brief LLVM pow transform
 * \author Josh Klontz \cite jklontz
 */
//class powTransform : public StitchableTransform
//{
//    Q_OBJECT
//    Q_PROPERTY(double exponent READ get_exponent WRITE set_exponent RESET reset_exponent STORED false)
//    BR_PROPERTY(double, exponent, 2)

//    int preallocate(const Matrix &src, Matrix &dst) const
//    {
//        dst.copyHeader(src);
//        dst.setFloating(true);
//        dst.setBits(max(src.bits(), 32));
//        return dst.elements();
//    }

//    Value *stitch(const MatrixBuilder &src, const MatrixBuilder &dst, Value *val) const
//    {
//        Value *load = src.cast(val, dst);

//        Value *pow;
//        if (exponent == ceil(exponent)) {
//            if      (exponent == 0) pow = dst.autoConstant(1);
//            else if (exponent == 1) pow = load;
//            else if (exponent == 2) pow = dst.multiply(load, load);
//            else                    pow = src.b->CreateCall2(Intrinsic::getDeclaration(TheModule, Intrinsic::powi, dst.tys()), load, MatrixBuilder::constant(int(exponent)));
//        } else {
//            pow = src.b->CreateCall2(Intrinsic::getDeclaration(TheModule, Intrinsic::pow, dst.tys()), load, dst.autoConstant(exponent));
//        }

//        return pow;
//    }
//};

//BR_REGISTER(Transform, powTransform)

/*!
 * \ingroup transforms
 * \brief LLVM sum transform
 * \author Josh Klontz \cite jklontz
 */
//class sumTransform : public UnaryTransform
//{
//    Q_OBJECT
//    Q_PROPERTY(bool channels READ get_channels WRITE set_channels RESET reset_channels STORED false)
//    Q_PROPERTY(bool columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
//    Q_PROPERTY(bool rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
//    Q_PROPERTY(bool frames READ get_frames WRITE set_frames RESET reset_frames STORED false)
//    BR_PROPERTY(bool, channels, true)
//    BR_PROPERTY(bool, columns, true)
//    BR_PROPERTY(bool, rows, true)
//    BR_PROPERTY(bool, frames, true)

//    int preallocate(const Matrix &src, Matrix &dst) const
//    {
//        dst = Matrix(channels ? 1 : src.channels, columns ? 1 : src.columns, rows ? 1 : src.rows, frames ? 1 : src.frames, src.hash);
//        dst.setBits(std::min(2*dst.bits(), dst.isFloating() ? 64 : 32));
//        return dst.elements();
//    }

//    Value *buildPreallocate(const MatrixBuilder &src, const MatrixBuilder &dst) const
//    {
//        (void) src;
//        (void) dst;
//        return MatrixBuilder::constant(0);
//    }

//    void build(const MatrixBuilder &src, const MatrixBuilder &dst, PHINode *i) const
//    {
//        Value *c, *x, *y, *t;
//        dst.deindex(i, &c, &x, &y, &t);
//        AllocaInst *sum = dst.autoAlloca(0, "sum");

//        QList<BasicBlock*> loops, exits;
//        loops.push_back(i->getParent());
//        Value *src_c, *src_x, *src_y, *src_t;

//        if (frames && !src.m->singleFrame()) {
//            BasicBlock *loop, *exit;
//            src_t = dst.beginLoop(loops.last(), loop, exit, src.frames(), "src_t");
//            loops.append(loop);
//            exits.append(exit);
//        } else {
//            src_t = t;
//        }

//        if (rows && !src.m->singleRow()) {
//            BasicBlock *loop, *exit;
//            src_y = dst.beginLoop(loops.last(), loop, exit, src.rows(), "src_y");
//            loops.append(loop);
//            exits.append(exit);
//        } else {
//            src_y = y;
//        }

//        if (columns && !src.m->singleColumn()) {
//            BasicBlock *loop, *exit;
//            src_x = dst.beginLoop(loops.last(), loop, exit, src.columns(), "src_x");
//            loops.append(loop);
//            exits.append(exit);
//        } else {
//            src_x = x;
//        }

//        if (channels && !src.m->singleChannel()) {
//            BasicBlock *loop, *exit;
//            src_c = dst.beginLoop(loops.last(), loop, exit, src.channels(), "src_c");
//            loops.append(loop);
//            exits.append(exit);
//        } else {
//            src_c = c;
//        }

//        dst.b->CreateStore(dst.add(dst.b->CreateLoad(sum), src.cast(src.load(src.aliasIndex(dst, src_c, src_x, src_y, src_t)), dst), "accumulate"), sum);

//        if (channels && !src.m->singleChannel()) dst.endLoop(loops.takeLast(), exits.takeLast());
//        if (columns && !src.m->singleColumn())   dst.endLoop(loops.takeLast(), exits.takeLast());
//        if (rows && !src.m->singleRow())         dst.endLoop(loops.takeLast(), exits.takeLast());
//        if (frames && !src.m->singleFrame())     dst.endLoop(loops.takeLast(), exits.takeLast());

//        dst.store(i, dst.b->CreateLoad(sum));
//    }
//};

//BR_REGISTER(Transform, sumTransform)

/*!
 * \ingroup transforms
 * \brief LLVM casting transform
 * \author Josh Klontz \cite jklontz
 */
//class castTransform : public StitchableTransform
//{
//    Q_OBJECT
//    Q_ENUMS(Type)
//    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)

//public:
//     /*!< */
//    enum Type { u1 = Matrix::u1,
//                u8 = Matrix::u8,
//                u16 = Matrix::u16,
//                u32 = Matrix::u32,
//                u64 = Matrix::u64,
//                s8 = Matrix::s8,
//                s16 = Matrix::s16,
//                s32 = Matrix::s32,
//                s64 = Matrix::s64,
//                f16 = Matrix::f16,
//                f32 = Matrix::f32,
//                f64 = Matrix::f64 };

//private:
//    BR_PROPERTY(Type, type, f32)

//    int preallocate(const Matrix &src, Matrix &dst) const
//    {
//        dst.copyHeader(src);
//        dst.setType(type);
//        return dst.elements();
//    }

//    Value *stitch(const MatrixBuilder &src, const MatrixBuilder &dst, Value *val) const
//    {
//        return src.cast(val, dst);
//    }
//};

//BR_REGISTER(Transform, castTransform)

/*!
 * \ingroup transforms
 * \brief LLVM scale transform
 * \author Josh Klontz \cite jklontz
 */
//class scaleTransform : public StitchableTransform
//{
//    Q_OBJECT
//    Q_PROPERTY(double a READ get_a WRITE set_a RESET reset_a STORED false)
//    BR_PROPERTY(double, a, 1)

//    Value *stitch(const MatrixBuilder &src, const MatrixBuilder &dst, Value *val) const
//    {
//        (void) src;
//        return dst.multiply(val, dst.autoConstant(a));
//    }
//};

//BR_REGISTER(Transform, scaleTransform)

/*!
 * \ingroup absTransform
 * \brief LLVM abs transform
 * \author Josh Klontz \cite jklontz
 */
//class absTransform : public StitchableTransform
//{
//    Q_OBJECT

//    Value *stitch(const MatrixBuilder &src, const MatrixBuilder &dst, Value *val) const
//    {
//        (void) dst;
//        if (!src.m->isSigned())  return val;
//        if (src.m->isFloating()) return src.b->CreateCall(Intrinsic::getDeclaration(TheModule, Intrinsic::fabs, src.tys()), val);
//        else                     return src.b->CreateSelect(src.b->CreateICmpSLT(val, src.autoConstant(0)),
//                                                            src.b->CreateSub(src.autoConstant(0), val),
//                                                            val);
//    }
//};

//BR_REGISTER(Transform, absTransform)

/*!
 * \ingroup transforms
 * \brief LLVM add transform
 * \author Josh Klontz \cite jklontz
 */
class addTransform : public StitchableTransform
{
    Q_OBJECT
    Q_PROPERTY(double b READ get_b WRITE set_b RESET reset_b STORED false)
    BR_PROPERTY(double, b, 0)

    Value *stitch(const MatrixBuilder &src, Value *val) const
    {
        return src.add(val, src.autoConstant(b));
    }
};

BR_REGISTER(Transform, addTransform)

/*!
 * \ingroup transforms
 * \brief LLVM clamp transform
 * \author Josh Klontz \cite jklontz
 */
//class clampTransform : public StitchableTransform
//{
//    Q_OBJECT
//    Q_PROPERTY(double min READ get_min WRITE set_min RESET reset_min STORED false)
//    Q_PROPERTY(double max READ get_max WRITE set_max RESET reset_max STORED false)
//    BR_PROPERTY(double, min, -std::numeric_limits<double>::max())
//    BR_PROPERTY(double, max, std::numeric_limits<double>::max())

//    Value *stitch(const MatrixBuilder &src, const MatrixBuilder &dst, Value *val) const
//    {
//        (void) src;
//        Value *clampedVal = val;
//        if (min > -std::numeric_limits<double>::max()) {
//            Value *low = dst.autoConstant(min);
//            clampedVal = dst.b->CreateSelect(dst.compareLT(clampedVal, low), low, clampedVal);
//        }
//        if (max < std::numeric_limits<double>::max()) {
//            Value *high = dst.autoConstant(max);
//            clampedVal = dst.b->CreateSelect(dst.compareGT(clampedVal, high), high, clampedVal);
//        }
//        return clampedVal;
//    }
//};

//BR_REGISTER(Transform, clampTransform)

/*!
 * \ingroup transforms
 * \brief LLVM quantize transform
 * \author Josh Klontz \cite jklontz
 */
//class _QuantizeTransform : public Transform
//{
//    Q_OBJECT
//    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a)
//    Q_PROPERTY(float b READ get_b WRITE set_b RESET reset_b)
//    BR_PROPERTY(float, a, 1)
//    BR_PROPERTY(float, b, 0)

//    QScopedPointer<Transform> transform;

//    void init()
//    {
//        transform.reset(Transform::make(QString("stitch([scale(%1),add(%2),clamp(0,255),cast(u8)])").arg(QString::number(a), QString::number(b))));
//    }

//    void train(const TemplateList &data)
//    {
//        (void) data;
//        qFatal("_Quantize::train not implemented.");
//    }

//    void project(const Template &src, Template &dst) const
//    {
//        transform->project(src, dst);
//    }
//};

// BR_REGISTER(Transform, _QuantizeTransform)

/*!
 * \ingroup initializers
 * \brief Initializes LLVM
 * \author Josh Klontz \cite jklontz
 */
class LLVMInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        QSharedPointer<Transform> kernel(Transform::make("add(1)", NULL));

        Template src, dst;
        src.m() = (Mat_<qint8>(2,2) << -1, -2, 3, 4);
        kernel->project(src, dst);
        qDebug() << dst.m();

        src.m() = (Mat_<qint32>(2,2) << -1, -3, 9, 27);
        kernel->project(src, dst);
        qDebug() << dst.m();

        src.m() = (Mat_<float>(2,2) << -1.5, -2.5, 3.5, 4.5);
        kernel->project(src, dst);
        qDebug() << dst.m();

        src.m() = (Mat_<double>(2,2) << 1.75, 2.75, -3.75, -4.75);
        kernel->project(src, dst);
        qDebug() << dst.m();
    }

    static void benchmark(const QString &transform)
    {
        static Template src;
        if (src.isEmpty()) {
            Mat m(4096, 4096, CV_32FC1);
            randu(m, 0, 255);
            src.m() = m;
        }

        QScopedPointer<Transform> original(Transform::make(transform, NULL));
        QScopedPointer<Transform> kernel(Transform::make(transform[0].toLower()+transform.mid(1), NULL));

        Template dstOriginal, dstKernel;
        original->project(src, dstOriginal);
        kernel->project(src, dstKernel);

        float difference = sum(dstKernel.m() - dstOriginal.m())[0]/(src.m().rows*src.m().cols);
        if (abs(difference) >= 0.0005)
            qWarning("Kernel result for %s differs by %.3f!", qPrintable(transform), difference);

        QTime time; time.start();
        for (int i=0; i<30; i++)
            kernel->project(src, dstKernel);
        double kernelTime = time.elapsed();

        time.start();
        for (int i=0; i<30; i++)
            original->project(src, dstOriginal);
        double originalTime = time.elapsed();

        qDebug("%s: %.3fx", qPrintable(transform), float(originalTime/kernelTime));
    }
};

BR_REGISTER(Initializer, LLVMInitializer)

likely_unary_function likely_make_unary_function(const char *description)
{
    QScopedPointer<UnaryTransform> unaryTransform(dynamic_cast<UnaryTransform*>(Transform::make(description, NULL)));
    if (unaryTransform == NULL) qFatal("makeUnaryFunction NULL transform!");
    return unaryTransform->getFunction();
}

likely_binary_function likely_make_binary_function(const char *description)
{
    (void) description;
    return NULL;
}

extern "C" {

LIKELY_EXPORT likely_unary_allocation likely_make_unary_allocation(const char *description, const likely_matrix *src)
{
    if (description == NULL) qFatal("makeUnaryAllocation NULL description!");
    const File f = long(description) < 1000 ? UnaryTransform::fileTable[long(description)] : File(description);
    QScopedPointer<UnaryTransform> unaryTransform(dynamic_cast<UnaryTransform*>(Transform::make(f, NULL)));
    if (unaryTransform == NULL) qFatal("makeUnaryKernel NULL transform!");
    return unaryTransform->getAllocation(src);
}

LIKELY_EXPORT likely_binary_allocation likely_make_binary_allocation(const char *description, const likely_matrix *src_a, const likely_matrix *src_b)
{
    (void) description;
    (void) src_a;
    (void) src_b;
    return NULL;
}

LIKELY_EXPORT likely_unary_kernel likely_make_unary_kernel(const char *description, const likely_matrix *src)
{
    if (description == NULL) qFatal("makeUnaryKernel NULL description!");
    const File f = long(description) < 1000 ? UnaryTransform::fileTable[long(description)] : File(description);
    QScopedPointer<UnaryTransform> unaryTransform(dynamic_cast<UnaryTransform*>(Transform::make(f, NULL)));
    if (unaryTransform == NULL) qFatal("makeUnaryKernel NULL transform!");
    return unaryTransform->getKernel(src);
}

LIKELY_EXPORT likely_binary_kernel likely_make_binary_kernel(const char *description, const likely_matrix *src_a, const likely_matrix *src_b)
{
    (void) description;
    (void) src_a;
    (void) src_b;
    return NULL;
}

} // extern "C"

#include "llvm.moc"
