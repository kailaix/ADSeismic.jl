using ADCME 

db = Database("simulation.db")

execute(db, """
CREATE TABLE IF NOT EXISTS acoustic 
(
    level text,
    size integer,
    forward real, 
    backward real,
    primary key (level, size)
)
""")

function insert_record(level, size, t1, t2)
    execute(db, """
    INSERT OR REPLACE into acoustic VALUES (\"$level\", $size, $t1, $t2)
    """)
    close(db)
end