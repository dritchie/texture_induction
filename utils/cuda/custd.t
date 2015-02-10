
local custd = {}

-- Copied from Terra's tests/cudaprintf.t
vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)
local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                local typ = e:gettype()
                local field = "_"..tonumber(i)
                typ = typ == float and double or typ
                table.insert(Buf.entries,{field,typ})
                emit quote
                   buf.[field] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end
custd.printf = macro(function(fmt,...)
    local buf = createbuffer({...})
    return `vprintf(fmt,buf) 
end)

custd.threadIdx = {}
custd.threadIdx.x = cudalib.nvvm_read_ptx_sreg_tid_x
custd.threadIdx.y = cudalib.nvvm_read_ptx_sreg_tid_y
custd.threadIdx.z = cudalib.nvvm_read_ptx_sreg_tid_z

custd.threadDim = {}
custd.threadDim.x = cudalib.nvvm_read_ptx_sreg_ntid_x
custd.threadDim.y = cudalib.nvvm_read_ptx_sreg_ntid_y
custd.threadDim.z = cudalib.nvvm_read_ptx_sreg_ntid_z

custd.blockIdx = {}
custd.blockIdx.x = cudalib.nvvm_read_ptx_sreg_ctaid_x
custd.blockIdx.y = cudalib.nvvm_read_ptx_sreg_ctaid_y
custd.blockIdx.z = cudalib.nvvm_read_ptx_sreg_ctaid_z

custd.blockDim = {}
custd.blockDim.x = cudalib.nvvm_read_ptx_sreg_nctaid_x
custd.blockDim.y = cudalib.nvvm_read_ptx_sreg_nctaid_y
custd.blockDim.z = cudalib.nvvm_read_ptx_sreg_nctaid_z

custd.warpSize = cudalib.nvvm_read_ptx_sreg_warpsize

custd.syncthreads = cudalib.nvvm_barrier0


return custd
