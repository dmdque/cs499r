"""load PLearn AMat files


An AMat file is an ascii format for dense matrices.

The format is not precisely defined, so I'll describe here a single recipe for making a valid
file.

.. code-block:: text

    #size: <rows> <cols>
    #sizes: <input cols> <target cols> <weight cols> <extra cols 0> <extra cols 1> <extra cols ...>
    number number number ....
    number number number ....


Tabs and spaces are both valid delimiters.  Newlines separate consecutive rows.

"""

import sys, numpy, array

class AMat:
    """DataSource to access a plearn amat file as a periodic unrandomized stream.

    Attributes:

    input -- all columns of input
    target -- all columns of target
    weight -- all columns of weight
    extra -- all columns of extra

    all -- the entire data contents of the amat file
    n_examples -- the number of training examples in the file

    AMat stands for Ascii Matri[x,ces]

    """

    marker_size = '#size:'
    marker_sizes = '#sizes:'
    marker_col_names = '#:'

    def __init__(self, path, head=None, update_interval=0, ofile=sys.stdout):

        """Load the amat at <path> into memory.

        path - str: location of amat file
        head - int: stop reading after this many data rows
        update_interval - int: print '.' to ofile every <this many> lines
        ofile - file: print status, msgs, etc. to this file

        """
        self.all = None
        self.input = None
        self.target = None
        self.weight = None
        self.extra = None

        self.header = False
        self.header_size = None
        self.header_rows = None
        self.header_cols = None
        self.header_sizes = None
        self.header_col_names = []

        data_started = False
        data = array.array('d')

        f = open(path)
        n_data_lines = 0
        len_float_line = None

        for i,line in enumerate(f):
            if n_data_lines == head:
                #we've read enough data,
                # break even if there's more in the file
                break
            if len(line) == 0 or line == '\n':
                continue
            if line[0] == '#':
                if not data_started:
                    #the condition means that the file has a header, and we're on
                    # some header line
                    self.header = True
                    if line.startswith(AMat.marker_size):
                        info = line[len(AMat.marker_size):]
                        self.header_size = [int(s) for s in info.split()]
                        self.header_rows, self.header_cols = self.header_size
                    if line.startswith(AMat.marker_col_names):
                        info = line[len(AMat.marker_col_names):]
                        self.header_col_names = info.split()
                    elif line.startswith(AMat.marker_sizes):
                        info = line[len(AMat.marker_sizes):]
                        self.header_sizes = [int(s) for s in info.split()]
            else:
                #the first non-commented line tells us that the header is done
                data_started = True
                float_line = [float(s) for s in line.split()]
                if len_float_line is None:
                    len_float_line = len(float_line)
                    if (self.header_cols is not None) \
                            and self.header_cols != len_float_line:
                        print >> sys.stderr, \
                                'WARNING: header declared %i cols but first line has %i, using %i',\
                                self.header_cols, len_float_line, len_float_line
                else:
                    if len_float_line != len(float_line):
                        raise IOError('wrong line length', i, line)
                data.extend(float_line)
                n_data_lines += 1

                if update_interval > 0 and (ofile is not None) \
                        and n_data_lines % update_interval == 0:
                    ofile.write('.')
                    ofile.flush()

        if update_interval > 0:
            ofile.write('\n')
        f.close()

        # convert from array.array to numpy.ndarray
        nshape = (len(data) / len_float_line, len_float_line)
        self.all = numpy.frombuffer(data).reshape(nshape)
        self.n_examples = self.all.shape[0]

        # assign
        if self.header_sizes is not None:
            if len(self.header_sizes) > 4:
                print >> sys.stderr, 'WARNING: ignoring sizes after 4th in %s' % path
            leftmost = 0
            #here we make use of the fact that if header_sizes has len < 4
            # the loop will exit before 4 iterations
            attrlist = ['input', 'target', 'weight', 'extra']
            for attr, ncols in zip(attrlist, self.header_sizes):
                setattr(self, attr, self.all[:, leftmost:leftmost+ncols])
                leftmost += ncols
