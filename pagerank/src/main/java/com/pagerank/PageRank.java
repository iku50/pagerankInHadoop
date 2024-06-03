package com.pagerank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 1,4 -> 1,1.0 -> 4,1.0 
 *
 */
public class PageRank
{
    public static int NODES = 281903;
    public static double d = 0.85;
    public static double r = (1-d)/(double)NODES;

    public static class Node {
        public int id;
        public double pageRank = 1.0;

        public Node() {
        }

        public Node(int id, double pageRank) {
            this.id = id;
            this.pageRank = pageRank;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("N:").append(id).append(";");
            sb.append(pageRank);
            return sb.toString();
        }

        public Node fromString(String str) {
            String[] parts = str.split(";");
            int id = Integer.parseInt(parts[0]);
            double pageRank = Double.parseDouble(parts[1]);
            return new Node(id, pageRank);
        }
    }
    
    public static class Edge {
        public int id;
        public List<Integer> edges = new ArrayList<Integer>();

        public Edge() {
        }
        public Edge(int id) {
            this.id = id;
        }

        // fromString method to create an Edge from a string
        public static Edge fromString(String s) {
            String[] parts = s.split(";");
            int id = Integer.parseInt(parts[0].trim());
            Edge edge = new Edge(id);
            if (parts.length > 1) {
                String[] edges = parts[1].split(",");
                for (String e : edges) {
                    edge.edges.add(Integer.parseInt(e));
                }
            }
            
            return edge;
        }

        // toString method to convert an Edge to a string
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("E:").append(id).append(";");
            for (int edgeId : edges) {
                sb.append(edgeId).append(",");
            }
            sb.deleteCharAt(sb.length()-1);
            return sb.toString();
        }
    }
    
    // first: collect all adj of from Node
    // a,b
    // a,d
    public static class PageRankMapper1 extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] nodes = value.toString().split("\t");
            Integer from = Integer.parseInt(nodes[0]);
            Integer to = Integer.parseInt(nodes[1]);
            context.write(new IntWritable(from), new IntWritable(to));
        }
    }
    
    // send 1/size(to Node) to all the toNode
    // and now we have fromNode's p value
    // so we need to sent it's p value to it self too.
    // a, a;0.5;b,d
    // b, a;0.5;b,d
    // d, a;0.5;b,d
    // data: fromNode: ToNodeList
    public static class PageRankReducer1 extends Reducer<IntWritable, IntWritable, IntWritable, Text> {
        public double p = 1.0;

        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            List<Integer> adList = new ArrayList<Integer>();
            for (IntWritable value : values) {
                adList.add(value.get());
            }
            Double pageRank = p * 1.0 / ((double) adList.size());
            // key is from Node
            Node node = new Node(key.get(), pageRank);
            for (Integer i : adList) {
                // i is toNode
                context.write(new IntWritable(i), new Text(node.toString()));
            }
            Edge e = new Edge(key.get());
            e.edges = adList;
            context.write(new IntWritable(key.get()), new Text(e.toString()));
        }
    }
    
    public static class PageRankMapper2 extends Mapper<LongWritable, Text, IntWritable, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] nodes = value.toString().split("\t");
            Integer i = Integer.parseInt(nodes[0]);
            context.write(new IntWritable(i), new Text(nodes[1]));
        }
    }
    
    // so the data will be: 
    // toNode: fromNodeList(fromNode's TonodeList) and it self's 
    // update pageRank
    
    public static class PageRankReducer2 extends Reducer<IntWritable, Text, IntWritable, Text> {
        
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double adding = 0.0;
            Edge itselfEdge = new Edge(key.get());
            Node itNode = new Node();
            Node node = new Node();
            for (Text value : values) {
                if (value.toString().startsWith("N:")) {
                    node = node.fromString(value.toString().replace("N:", ""));
                    adding += node.pageRank;
                } else {
                    itselfEdge = Edge.fromString(value.toString().replace("E:", ""));
                }
            }
            itNode.id = itselfEdge.id;
            itNode.pageRank = (d * adding + r) / (double) itselfEdge.edges.size();
            for (Integer i : itselfEdge.edges) {
                context.write(new IntWritable(i), new Text(itNode.toString()));
            }
            context.write(key, new Text(itselfEdge.toString()));
        }
    }

    public static class PageRankMapper3 extends Mapper<LongWritable, Text, IntWritable, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] nodes = value.toString().split("\t");
            Integer i = Integer.parseInt(nodes[0]);
            context.write(new IntWritable(i), new Text(nodes[1]));
        }
    }
    // so the data will be: 
    // toNode: fromNodeList(fromNode's TonodeList) and it self's 
    // update pageRank

    public static class PageRankReducer3 extends Reducer<IntWritable, Text, IntWritable, Text> {
        
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double adding = 0.0;
            Node itNode = new Node();
            Node node = new Node();
            
            for (Text value : values) {
                if (value.toString().startsWith("N:")) {
                    node = node.fromString(value.toString().replace("N:", ""));
                    adding += node.pageRank;
                }
            }
            itNode.id = key.get();    
            itNode.pageRank = d * adding + r;

            // for (Integer i : itselfEdge.edges) {
            //     context.write(new IntWritable(i), new Text(itNode.toString()));
            // }
            context.write(key, new Text(itNode.toString()));
        }
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator", ",");
        Path inputPath = new Path(args[0]);
        Path middlePath = new Path(args[1]);
        Path middle2Path = new Path(args[2]);
        Path outputPath = new Path(args[3]);
        
        Job job = Job.getInstance(conf, "PageRank");
        job.setJarByClass(PageRank.class);
        job.setMapperClass(PageRankMapper1.class);
        job.setReducerClass(PageRankReducer1.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, middlePath);
        if (!job.waitForCompletion(true)) {
            System.exit(1);
        }

        int iteration = 0;
        int iterationLimit = 10;  
        boolean status = false;
        Path onePath = middlePath;
        Path twoPath = middle2Path;

        while (iteration < iterationLimit) {
            twoPath = new Path(middle2Path.toString() + String.valueOf(iteration));
            Configuration conf2 = new Configuration();

            Job job2 = Job.getInstance(conf2, "PageRank2");
            job2.setJarByClass(PageRank.class);
            job2.setMapperClass(PageRankMapper2.class);
            job2.setReducerClass(PageRankReducer2.class);
            job2.setOutputKeyClass(IntWritable.class);
            job2.setOutputValueClass(Text.class);
            FileInputFormat.addInputPath(job2, onePath);
            FileOutputFormat.setOutputPath(job2, twoPath);
            status = job2.waitForCompletion(true);
            if (!status) {
                System.exit(1);
            }
            iteration++;  
            onePath = twoPath; 
        }  

        Configuration conf3 = new Configuration();

        Job job3 = Job.getInstance(conf3, "PageRank3");
        job3.setJarByClass(PageRank.class);
        job3.setMapperClass(PageRankMapper3.class);
        job3.setReducerClass(PageRankReducer3.class);
        job3.setOutputKeyClass(IntWritable.class);
        job3.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job3, twoPath);
        FileOutputFormat.setOutputPath(job3, outputPath);
        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}
