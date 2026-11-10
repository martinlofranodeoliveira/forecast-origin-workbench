# Security boundaries

The package accepts numerical series and closed local JSON, not model files,
expressions, plugins, remote endpoints or credentials. The CLI opens bounded
regular snapshots with no-follow/nonblocking leaf handling; this is a cooperative
filesystem contract, not a hostile-parent sandbox. Stdin requires its producer to
finish; a byte cap is not an OS deadline. Native library initialization and
allocations require an outer resource limit when that matters.

JSON reports contain actual values/predictions and identifying hashes. Treat them
like the source data. Error diagnostics omit input values/paths and dependency
exception text. Report suspected defects through the repository's private
vulnerability-reporting channel when available; examples should be synthetic.
