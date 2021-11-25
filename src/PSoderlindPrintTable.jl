"""

Paul Soderlind printTable and printmat: functions for nicer table and matrix printing.

Downloaded from https://github.com/PaulSoderlind/JuliaTutorial, June 2020


printTable
printmat
printlnPs
printmat2


"""

using Dates


"""
    printTable([fh::IO],x,colNames=[],rowNames=[];
               width=10,prec=3,NoPrinting=false,htmlQ=false,cell00="")
Print formatted table with row names (1st column) column names (1st row), and data matrix (the rest).
# Input
- `fh::IO`:           (optional) file handle. If not supplied, prints to screen
- `x::Array`:         string, date or array to print
- `width::Int`:       (keyword) scalar, width of printed cells. [10]
- `prec::Int`:        (keyword) scalar, precision of printed cells. [3]
- `NoPrinting::Bool`: (keyword) bool, true: no printing, just return formatted string [false]
- `hmtlQ::Bool`:      (keyword) bool, true: format as htmlQ <td>cells</td> [false]
- `cell00::String`:   (keyword) string, for row 0, column 0
# Output
- `str::String`:      (if NoPrinting) string, (otherwise nothing)
# Example
```
xA = [1 "ab" "abc"; "ccc" 3.14 missing]
printTable(xA,colNames,["1";"4"],width=12,prec=2)
```
# Uses
- printmat
"""
function printTable(fh::IO,x,colNames=[],rowNames=[];
                    width=10,prec=3,NoPrinting=false,htmlQ=false,cell00="")

  isempty(x) && return nothing                        #do nothing is isempty(x)

  (m,n) = (size(x,1),size(x,2))

  if isempty(rowNames)                                 #create row names "r1"
    rowNames = [string("r",i) for i = 1:m]
  end
  if isempty(colNames)                                 #create column names "c1"
    colNames = [string("c",i) for i = 1:n]
  end

  rNamesWidth = maximum([length(rowNames[i]) for i = 1:length(rowNames)])  #max length of rowNames
  rNamesWidth = max(rNamesWidth,length(cell00))

  if htmlQ                                             #print column names
    cNamesStr = string("<tr><th>",lpad(cell00,rNamesWidth),"</th>")
    for i = 1:n
      cNamesStr = string(cNamesStr,"<th>",lpad(colNames[i],width),"</th>")
    end
    cNamesStr = string(cNamesStr,"</tr>")
  else
    cNamesStr = lpad(cell00,rNamesWidth)                 #cell 0,0
    for i = 1:n                                          #create string
      cNamesStr = string(cNamesStr,lpad(colNames[i],width))
    end
  end

  xStr  = printmat(fh,x,width=width,prec=prec,NoPrinting=true,htmlQ=htmlQ)   #body of table, one long string
  xStrV = split(xStr,"\n")                       #vector of strings (one per row of x)

  iob = IOBuffer()
  write(iob,cNamesStr,"\n")
  for i = 1:m                           #loop over rows in x, print rowNames[i] and x[i,:]
    if htmlQ
      write(iob,"<tr><td><b>",rpad(rowNames[i],rNamesWidth),"</td></b>",xStrV[i],"</tr> \n")
    else
      write(iob,rpad(rowNames[i],rNamesWidth),xStrV[i],"\n")
    end
  end
  str = String(take!(iob))

  if NoPrinting                              #no printing, just return str
    return str
  else                                       #print, return nothing
    print(fh,str,"\n")
    return nothing
  end

end
                        #when fh is not supplied: printing to screen
printTable(x,colNames=[],rowNames=[];width=10,prec=3,NoPrinting=false,htmlQ=false,cell00="") =
printTable(stdout::IO,x,colNames,rowNames,
                      width=width,prec=prec,NoPrinting=NoPrinting,htmlQ=htmlQ,cell00=cell00)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    printTable2
Call on printTable2 twice: to print to screen and then to an open file (IOStream)
"""
function printTable2(fh,x,colNames=[],rowNames=[];width=10,prec=3,NoPrinting=false,htmlQ=false,cell00="")
  printTable(x,colNames,rowNames,width=width,prec=prec,NoPrinting=NoPrinting,htmlQ=htmlQ,cell00=cell00)      #to screen
  if isa(fh,IOStream) && isopen(fh)
    printTable(fh,x,colNames,rowNames,width=width,prec=prec,NoPrinting=NoPrinting,htmlQ=htmlQ,cell00=cell00) #to file
  end
end
#------------------------------------------------------------------------------







#------------------------------------------------------------------------------
"""
    printmat([fh::IO],x;width=10,prec=3,NoPrinting=false,htmlQ=false)
Print all elements of matrix with predefined formatting.
# Input
- `fh::IO`:           (optional) file handle. If not supplied, prints to screen
- `x::Array`:         string, date or array to print
- `width::Int`:       (keyword) scalar, width of printed cells. [10]
- `prec::Int`:        (keyword) scalar, precision of printed cells. [3]
- `NoPrinting::Bool`: (keyword) bool, true: no printing, just return formatted string [false]
- `hmtlQ::Bool`:      (keyword) bool, true: format as htmlQ <td>cells</td> [false]
# Output
- str         (if NoPrinting) string, (otherwise nothing)
# Examples. Try printing the following arrays:
- x = [11 12;21 22]
- x = Any[1 "ab"; Date(2018,10,7) 3.14]
# Uses
- fmtNumPs
# To do
- use Dict() for the options, width etc?
Paul.Soderlind@unisg.ch
"""
function printmat(fh::IO,x;width=10,prec=3,NoPrinting=false,htmlQ=false)

  if isa(x,Union{String,Date,DateTime,Missing})  #eg. a single Date
    str = string(lpad(x,width),"\n")
    if NoPrinting
      return str
    else
      print(fh,str,"\n")
      return nothing
    end
  elseif isa(x,Nothing)
    return nothing
  end

  if ndims(x) > 2
    @warn("more than 2 dimensions")
    return nothing
  end

  (m,n) = (size(x,1),size(x,2))

  iob = IOBuffer()
  for i = 1:m                #loop over lines
    for j = 1:n                #loop over columns
      if isa(x[i,j],AbstractFloat)        #Float
        write(iob,fmtNumPs(x[i,j],width,prec,"right",htmlQ))
      elseif isa(x[i,j],Bool)             #Bool, BitArrays
        htmlQ ? write(iob,"<td>",lpad(x[i,j]+0,width),"</td>") : write(iob,lpad(x[i,j]+0,width))
      elseif isa(x[i,j],Nothing)          #Nothing
        htmlQ ? write(iob,"<td>",lpad("",width),"</td>") : write(iob,lpad("",width))
      elseif isa(x[i,j],String)           #String, left justified
        htmlQ ? write(iob,"<td>",rpad(x[i,j],width),"</td>") : write(iob,rpad(x[i,j],width))
      else                                #other types (Integer,Missing,Date,...)
        htmlQ ? write(iob,"<td>",lpad(x[i,j],width),"</td>") : write(iob,lpad(x[i,j],width))
      end
    end
    write(iob,"\n")            #newline
  end
  str = String(take!(iob))

  if NoPrinting                              #no printing, just return str
    return str
  else                                       #print, return nothing
    print(fh,str,"\n")
    return nothing
  end

end
                  #when fh is not supplied: printing to screen
printmat(x;width=10,prec=3,NoPrinting=false,htmlQ=false) = printmat(stdout::IO,
         x,width=width,prec=prec,NoPrinting=NoPrinting,htmlQ=htmlQ)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    fmtNumPs(z,width=10,prec=2,justify="right",htmlQ=false)
Formats a scalar and creates a string of it.
# Remark
The Formatting.jl package provides more elegant solutions:
fmt  = FormatSpec(string(">",width,".",prec,"f"))   #right justified, else "<"
fmt  = FormatSpec(string(">",wid,"d"))              #for Int
str  = Formatting.fmt(fmt1,z))
"""
function fmtNumPs(z,width=10,prec=2,justify="right",htmlQ=false)
  if prec > 0                        #if decimal number
    z   = round(z,digits=prec)       #101.23
    str = split(string(z),'.')
    if length(str) > 1
      strR  = string(".",rpad(str[2],prec,"0"))   #.23
      strLR = string(str[1],strR)                 #"101" * ".23"
    else                                          #eg. NaN, missing
      strLR = string(z)
    end
  else
    if typeof(z) <: AbstractFloat                  #Floats
      z = round(Int,z)
    end
    strLR = string(z)
  end
  if justify == "left"
    strLR = rpad(strLR,width)
  else
    strLR = lpad(strLR,width)
  end
  htmlQ && (strLR = string("<td>",strLR,"</td>"))
  return strLR
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    printlnPs([fh::IO],z...;width=10,prec=3)
Subsitute for println, with predefined formatting.
# Input
- `fh::IO`:    (optional) file handle. If not supplied, prints to screen
- `z::String`: string, numbers and arrays to print
Paul.Soderlind@unisg.ch
"""
function printlnPs(fh::IO,z...;width=10,prec=3)

  for x in z                              #loop over inputs in z...
    if isa(x,Union{String,Date,DateTime,Missing})
      print(fh,lpad(x,width))
    elseif isa(x,Nothing)
      print(fh,"")
    else                                         #other types
      iob = IOBuffer()
      for i = 1:length(x)
        if isa(x[i],AbstractFloat)               #Float
          write(iob,fmtNumPs(x[i],width,prec,"right"))
        elseif isa(x[i],Nothing)                 #Nothing
          write(iob,lpad("",width))
        else                                     #Integer, etc
          write(iob,lpad(x[i],width))
        end
      end
      print(fh,String(take!(iob)))
    end
  end

  print(fh,"\n")

end
                      #when fh is not supplied: printing to screen
printlnPs(z...;width=10,prec=3) = printlnPs(stdout::IO,z...,width=width,prec=prec)
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    println2Ps
Call on printlnPs twice: to print to screen and then to an open file (IOStream)
"""
function println2Ps(fh::IO,z...;width=10,prec=3)
  printlnPs(z...,width=width,prec=prec)              #to screen
  if isa(fh,IOStream) && isopen(fh)
    printlnPs(fh::IO,z...,width=width,prec=prec)     #to file
  end
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
"""
    printmat2
Call on printmat twice: to print to screen and then to an open file (IOStream)
"""
function printmat2(fh,x;width=10,prec=3,NoPrinting=false,htmlQ=false)
  printmat(x,width=width,prec=prec,NoPrinting=NoPrinting,htmlQ=htmlQ)       #to screen
  if isa(fh,IOStream) && isopen(fh)
    printmat(fh,x,width=width,prec=prec,NoPrinting=NoPrinting,htmlQ=htmlQ)  #to file
  end
end
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
function printblue(x...)
  foreach(z->printstyled(z,color=:blue,bold=true),x)
  print("\n")
end
function printred(x...)
  foreach(z->printstyled(z,color=:red,bold=true),x)
  print("\n")
end
function printmagenta(x...)
  foreach(z->printstyled(z,color=:magenta,bold=true),x)
  print("\n")
end
function printyellow(x...)
  foreach(z->printstyled(z,color=:yellow,bold=true),x)
  print("\n")
end
#------------------------------------------------------------------------------
